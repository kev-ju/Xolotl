// tools/v41/engine/ds41_chunk_cpux.hpp - STRATA_CHUNK_CPUX: a prompt chunk's streamed experts computed on the CPU with
// the bits llama.cpp's MMQ gives them on the card (ds41_moe.cpp, cpux_post / cpux_collect).
//
// WHAT IS REPRODUCED.  run_chunk multiplies a streamed expert with MMQ (mmq.cuh, the int8 tensor-core kernels):
// the activations are block_q8_1_mmq rows (q8, one float scale per 32 values: the D4 layout) and a Q2_0 weight row is
// int8 values code - 1 with one fp16 scale per 64.  For output element (row i, column j) the Ampere kernel
// (ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma) starts from +0.0f and adds, for the row's 32-value chunks c in ascending order,
//
//     sum += (float) C_c * dA_c * dB_c          C_c = the chunk's integer dot product, dA_c the weight block's scale,
//                                               dB_c the activation chunk's scale
//
// With one CUDA block per tile (tools/v41/ggml/mmq-deterministic.patch: no stream-k split for the MoE products) that
// order does not depend on the tile width or on which experts share a launch.  nvcc contracts the last product and the
// add (-fmad, its default): sum = fma((float) C_c * dA_c, dB_c, sum).  `fma` = false keeps them apart (a build with
// -fmad=false); STRATA_CHUNK_CPUX_CHECK compares every row with the card's and says which one this build needs.
// Integer dot products are exact, and every float operation here is a single IEEE operation, so the rows are the
// card's bit for bit.  PTQ1_0 (ternary) matrices are multiplied as the Q2_0 copy ds41_transcode makes of them: the
// same d, value = trit.
//
// WHAT IS NOT.  The activation quantizers and the SwiGLU (its __expf) stay on the card: the caller runs those very
// kernels on the rows and hands their bytes over.  Only the three products - the part that reads the weights - are
// computed here.
//
// x86-64 with AVX2 + FMA + F16C (cpux::cpu_ok()); header-only so the engine's build files do not change.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#define DS41_CPUX_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define DS41_CPUX_TARGET
#else
#include <cpuid.h>
#define DS41_CPUX_TARGET __attribute__((target("avx2,fma,f16c")))
#endif
#endif

namespace strata::v41::cpux {

constexpr int kTypeQ20 = 42;     ///< GGML Q2_0: 18 B per 64 weights {fp16 d, 16 B of 2-bit codes}, value = code - 1
constexpr int kTypePtq1 = 143;   ///< PTQ1_0: 28 B per 128 weights {qs[24], qh[2] base-3 digits, fp16 d}, value = digit - 1
constexpr int kTypeQ2K = 10;     ///< GGML Q2_K: 84 B per 256 weights {scales[16] (scale | min << 4), qs[64], fp16 d, fp16 dmin}
constexpr int kQ81Mmq = 144;     ///< sizeof(block_q8_1_mmq): float d4[4], int8 qs[128]
constexpr int kMaxK = 8192;      ///< the widest weight row (values) the scratch holds
constexpr int kTokBatch = 8;     ///< activations one sweep over a row group carries

inline bool type_ok(int t) { return t == kTypeQ20 || t == kTypePtq1 || t == kTypeQ2K; }
inline size_t row_bytes(int type, int K) {
    return type == kTypeQ20 ? (size_t) (K / 64) * 18 : type == kTypeQ2K ? (size_t) (K / 256) * 84 : (size_t) (K / 128) * 28;
}
/// The activation layout MMQ multiplies a weight type with: 0 = D4 (one float scale per 32 values; Q2_0 and the PTQ1_0
/// copies), 1 = D2S6 (Q2_K: one fp16 scale per 64 values and fp16 partial sums of the unquantized values per 16).
inline int act_kind(int type) { return type == kTypeQ2K ? 1 : 0; }

inline float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31u;
    uint32_t m = h & 1023u, f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            uint32_t sh = 0;
            while (!(m & 1024u)) { m <<= 1; ++sh; }
            f = s | ((113u - sh) << 23) | ((m & 1023u) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

/// Activation rows as the card quantized them (mmq::quantize, the D4 layout: block b of row r at (b * rows + r) * 144),
/// repacked per row: q[K] int8, one scale and one sum of q per 32 values.
struct Acts {
    int K = 0, n = 0, nc = 0;
    std::vector<int8_t> q;
    std::vector<float> db;
    std::vector<int32_t> sq;
    void build(const uint8_t* y, int64_t rows, int K_) {
        K = K_;
        n = (int) rows;
        nc = K / 32;
        q.resize((size_t) n * (size_t) K + 64);
        db.resize((size_t) n * (size_t) nc);
        sq.resize((size_t) n * (size_t) nc);
        const int nb = K / 128;
        for (int r = 0; r < n; ++r) {
            int8_t* qr = q.data() + (size_t) r * (size_t) K;
            float* dr = db.data() + (size_t) r * (size_t) nc;
            int32_t* sr = sq.data() + (size_t) r * (size_t) nc;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* src = y + ((size_t) b * (size_t) rows + (size_t) r) * kQ81Mmq;
                std::memcpy(dr + 4 * b, src, 16);
                std::memcpy(qr + 128 * b, src + 16, 128);
                for (int c = 0; c < 4; ++c) {
                    int32_t s = 0;
                    const int8_t* p = qr + 128 * b + 32 * c;
                    for (int i = 0; i < 32; ++i) s += p[i];
                    sr[4 * b + c] = s;
                }
            }
        }
    }
    const int8_t* qp(int r) const { return q.data() + (size_t) r * (size_t) K; }
    const float* dbp(int r) const { return db.data() + (size_t) r * (size_t) nc; }
    const int32_t* sqp(int r) const { return sq.data() + (size_t) r * (size_t) nc; }
};

/// The same for the D2S6 layout (Q2_K weights): per 128-value block half d[2] (one per 64 values) and half s[6] (the
/// sum of the unquantized values of each of the first six 16-value groups), then q[128]; plus the sum of q per 16.
struct ActsK {
    int K = 0, n = 0, nb = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;      // n x nb x 2, n x nb x 6
    std::vector<int32_t> sq;      // n x K / 16
    void build(const uint8_t* y, int64_t rows, int K_) {
        K = K_;
        n = (int) rows;
        nb = K / 128;
        q.resize((size_t) n * (size_t) K + 64);
        d.resize((size_t) n * (size_t) nb * 2);
        s.resize((size_t) n * (size_t) nb * 6);
        sq.resize((size_t) n * (size_t) (K / 16));
        for (int r = 0; r < n; ++r) {
            int8_t* qr = q.data() + (size_t) r * (size_t) K;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* src = y + ((size_t) b * (size_t) rows + (size_t) r) * kQ81Mmq;
                uint16_t h[8];
                std::memcpy(h, src, 16);
                for (int i = 0; i < 2; ++i) d[((size_t) r * nb + b) * 2 + i] = half_to_float(h[i]);
                for (int i = 0; i < 6; ++i) s[((size_t) r * nb + b) * 6 + i] = half_to_float(h[2 + i]);
                std::memcpy(qr + 128 * b, src + 16, 128);
                for (int c = 0; c < 8; ++c) {
                    int32_t t = 0;
                    const int8_t* p = qr + 128 * b + 16 * c;
                    for (int i = 0; i < 16; ++i) t += p[i];
                    sq[(size_t) r * (size_t) (K / 16) + 8 * b + c] = t;
                }
            }
        }
    }
    const int8_t* qp(int r) const { return q.data() + (size_t) r * (size_t) K; }
    const float* dp(int r) const { return d.data() + (size_t) r * (size_t) nb * 2; }
    const float* sp(int r) const { return s.data() + (size_t) r * (size_t) nb * 6; }
    const int32_t* sqp(int r) const { return sq.data() + (size_t) r * (size_t) (K / 16); }
};

#if defined(DS41_CPUX_X86)

inline bool cpu_ok() {
    static const bool ok = [] {
#if defined(_MSC_VER) && !defined(__clang__)
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const bool osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1;
        if (!(osx && avx && fma && f16c)) return false;
        if ((_xgetbv(0) & 6) != 6) return false;
        __cpuidex(r, 7, 0);
        return ((r[1] >> 5) & 1) != 0;   // AVX2
#else
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
    }();
    return ok;
}

/// One thread's scratch: 8 decoded weight rows (codes 0..3, value = code - 1) and their chunk scales, chunk-major.
struct Scratch {
    std::vector<uint8_t> u;     // 8 x (kMaxK + 64)
    std::vector<float> da;      // (kMaxK / 32) x 8; Q2_K: (kMaxK / 16) x 8 scales
    std::vector<float> ma;      // Q2_K: (kMaxK / 16) x 8 mins
    Scratch() : u((size_t) 8 * (kMaxK + 64)), da((size_t) (kMaxK / 16) * 8), ma((size_t) (kMaxK / 16) * 8) {}
    uint8_t* row(int i) { return u.data() + (size_t) i * (kMaxK + 64); }
};

inline const uint32_t* q2_lut() {
    static const struct T {
        uint32_t v[256];
        T() {
            for (uint32_t b = 0; b < 256; ++b)
                v[b] = (b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) | (((b >> 6) & 3u) << 24);
        }
    } t;
    return t.v;
}

DS41_CPUX_TARGET inline float f16_to_f32(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

/// One PTQ1_0 block's 128 digits (0, 1, 2 = trit -1, 0, +1 = the Q2_0 code ds41_transcode writes) into u[0, 128).
/// Element 16 t + m is digit t of qs[m] (m < 16), 80 + 8 t + m digit t of qs[16 + m] (m < 8), 120 + 2 t + h digit t
/// of qh[h]; digit t of byte b is (((b * 3^t) & 255) * 3) >> 8  (ds41_transcode.cu's ptq1_digit).
DS41_CPUX_TARGET inline void ptq1_block(const uint8_t* blk, uint8_t* u) {
    static const short p3[5] = { 1, 3, 9, 27, 81 };
    const __m256i w16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) blk));
    const __m256i lo16 = _mm256_set1_epi16(0x00FF), three16 = _mm256_set1_epi16(3);
    for (int t = 0; t < 5; ++t) {
        const __m256i x = _mm256_and_si256(_mm256_mullo_epi16(w16, _mm256_set1_epi16(p3[t])), lo16);
        const __m256i d = _mm256_srli_epi16(_mm256_mullo_epi16(x, three16), 8);
        const __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(d, d), 0xD8);
        _mm_storeu_si128((__m128i*) (u + 16 * t), _mm256_castsi256_si128(pk));
    }
    const __m128i w8 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (blk + 16)));
    const __m128i lo8 = _mm_set1_epi16(0x00FF), three8 = _mm_set1_epi16(3);
    for (int t = 0; t < 5; ++t) {
        const __m128i x = _mm_and_si128(_mm_mullo_epi16(w8, _mm_set1_epi16(p3[t])), lo8);
        const __m128i d = _mm_srli_epi16(_mm_mullo_epi16(x, three8), 8);
        _mm_storel_epi64((__m128i*) (u + 80 + 8 * t), _mm_packus_epi16(d, d));
    }
    for (int h = 0; h < 2; ++h) {
        uint32_t x = blk[24 + h];
        for (int t = 0; t < 4; ++t) {
            u[120 + 2 * t + h] = (uint8_t) ((x * 3u) >> 8);
            x = (x * 3u) & 0xFFu;
        }
    }
}

/// Decodes one weight row of K values: codes into u[0, K), the scale of each 32-value chunk into da[c * da_stride].
DS41_CPUX_TARGET inline void decode_row(int type, const uint8_t* w, int K, uint8_t* u, float* da, int da_stride) {
    if (type == kTypeQ20) {
        const uint32_t* lut = q2_lut();
        const int nb = K / 64;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = w + (size_t) b * 18;
            const float d = f16_to_f32(blk);
            da[(size_t) (2 * b) * da_stride] = d;
            da[(size_t) (2 * b + 1) * da_stride] = d;
            uint8_t* o = u + 64 * b;
            for (int i = 0; i < 16; ++i) std::memcpy(o + 4 * i, &lut[blk[2 + i]], 4);
        }
    } else {
        const int nb = K / 128;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = w + (size_t) b * 28;
            const float d = f16_to_f32(blk + 26);
            for (int c = 0; c < 4; ++c) da[(size_t) (4 * b + c) * da_stride] = d;
            ptq1_block(blk, u + 128 * b);
        }
    }
}

/// The sums of eight vectors of eight int32: lane i = the sum of v[i].
DS41_CPUX_TARGET inline __m256i hsum8(const __m256i* v) {
    const __m256i s01 = _mm256_hadd_epi32(v[0], v[1]), s23 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i s45 = _mm256_hadd_epi32(v[4], v[5]), s67 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i a = _mm256_hadd_epi32(s01, s23), b = _mm256_hadd_epi32(s45, s67);
    const __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i hi = _mm_add_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/// Eight decoded weight rows (sc.row(i), sc.da) against `nt` activation rows A[aidx[t]]: out[t][0..8) = the MMQ sums
/// of the eight rows for activation t.
DS41_CPUX_TARGET inline void group8(Scratch& sc, int K, const Acts& A, const int32_t* aidx, int nt, float* const* out,
                                    bool fma) {
    const __m256i ones = _mm256_set1_epi16(1);
    const int nc = K / 32;
    const uint8_t* ur[8];
    for (int i = 0; i < 8; ++i) ur[i] = sc.row(i);
    const float* da = sc.da.data();
    for (int t0 = 0; t0 < nt; t0 += kTokBatch) {
        const int tn = std::min(kTokBatch, nt - t0);
        __m256 acc[kTokBatch];
        const int8_t* q[kTokBatch];
        const float* db[kTokBatch];
        const int32_t* sq[kTokBatch];
        for (int i = 0; i < tn; ++i) {
            acc[i] = _mm256_setzero_ps();
            q[i] = A.qp(aidx[t0 + i]);
            db[i] = A.dbp(aidx[t0 + i]);
            sq[i] = A.sqp(aidx[t0 + i]);
        }
        for (int c = 0; c < nc; ++c) {
            __m256i u[8];
            for (int j = 0; j < 8; ++j) u[j] = _mm256_loadu_si256((const __m256i*) (ur[j] + 32 * c));
            const __m256 dav = _mm256_loadu_ps(da + 8 * c);
            for (int i = 0; i < tn; ++i) {
                const __m256i qv = _mm256_loadu_si256((const __m256i*) (q[i] + 32 * c));
                __m256i v[8];
                for (int j = 0; j < 8; ++j) v[j] = _mm256_madd_epi16(_mm256_maddubs_epi16(u[j], qv), ones);
                // sum(code * q) - sum(q) = sum((code - 1) * q): the tile's int8 values are code - 1
                const __m256i dot = _mm256_sub_epi32(hsum8(v), _mm256_set1_epi32(sq[i][c]));
                const __m256 tm = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), dav);   // (float) C * dA
                const __m256 dbv = _mm256_broadcast_ss(db[i] + c);
                acc[i] = fma ? _mm256_fmadd_ps(tm, dbv, acc[i]) : _mm256_add_ps(acc[i], _mm256_mul_ps(tm, dbv));
            }
        }
        for (int i = 0; i < tn; ++i) _mm256_storeu_ps(out[t0 + i], acc[i]);
    }
}

// ---- Q2_K (ggml_cuda_mmq_load_tiles_q2_K / ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma, the Ampere branch) ----
// The tile holds the 2-bit codes as they are (0..3) and, per 16-value group, the fp16 products d * scale and
// dmin * min (__hmul2: one fp16 multiply each).  For element (i, j) and each 128-value block, in this order:
//   for the four 32-value chunks c:   tmp  = C0 * dA0 + C1 * dA1            (the chunk's two groups)
//                                     [c == 3:  tmp -= S0 * mA0 + S1 * mA1,  S = the sum of q over the group]
//                                     sum += tmp * dB                        (dB: the fp16 scale of the chunk's 64)
//   for the first six groups g:       sum -= mA_g * sB_g                     (sB: the fp16 sum of the unquantized values)
// `var` picks how the compiler contracted them: bit 0 = no fused multiply-add at all; bit 1 / bit 2 = the second
// product of tmp / of the min term is the fused one; bit 3 = the last line not fused.

DS41_CPUX_TARGET inline float hmul_to_f32(float a, float b) {   // fp16(a * b) as a float; a, b hold fp16 values
    const __m128i h = _mm_cvtps_ph(_mm_set_ss(a * b), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_cvtss_f32(_mm_cvtph_ps(h));
}

DS41_CPUX_TARGET inline __m256 mul_add(bool nofma, __m256 a, __m256 b, __m256 c) {   // a * b + c
    return nofma ? _mm256_add_ps(_mm256_mul_ps(a, b), c) : _mm256_fmadd_ps(a, b, c);
}

/// One Q2_K weight row: codes into u[0, K), the 16-value groups' scale and min products into da / ma (stride apart).
DS41_CPUX_TARGET inline void decode_row_q2k(const uint8_t* w, int K, uint8_t* u, float* da, float* ma, int stride) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const int nb = K / 256;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = w + (size_t) b * 84;
        const float d = f16_to_f32(blk + 80), dmin = f16_to_f32(blk + 82);
        for (int g = 0; g < 16; ++g) {
            const int sc = blk[g];
            da[(size_t) (16 * b + g) * stride] = hmul_to_f32(d, (float) (sc & 15));
            ma[(size_t) (16 * b + g) * stride] = hmul_to_f32(dmin, (float) (sc >> 4));
        }
        for (int h = 0; h < 2; ++h) {
            const __m256i v = _mm256_loadu_si256((const __m256i*) (blk + 16 + 32 * h));
            for (int l = 0; l < 4; ++l)
                _mm256_storeu_si256((__m256i*) (u + 256 * b + 128 * h + 32 * l),
                                    _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * l)), m3));
        }
    }
}

DS41_CPUX_TARGET inline void group8k(Scratch& sc, int K, const ActsK& A, const int32_t* aidx, int nt, float* const* out,
                                     int var) {
    const bool nofma = (var & 1) != 0, sw1 = (var & 2) != 0, sw2 = (var & 4) != 0, nof3 = (var & 8) != 0;
    const __m256i ones = _mm256_set1_epi16(1);
    const int nb = K / 128;
    const uint8_t* ur[8];
    for (int i = 0; i < 8; ++i) ur[i] = sc.row(i);
    const float* da = sc.da.data();
    const float* ma = sc.ma.data();
    for (int t0 = 0; t0 < nt; t0 += kTokBatch) {
        const int tn = std::min(kTokBatch, nt - t0);
        __m256 acc[kTokBatch];
        const int8_t* q[kTokBatch];
        const float* dq[kTokBatch];
        const float* sq[kTokBatch];
        const int32_t* s16[kTokBatch];
        for (int i = 0; i < tn; ++i) {
            acc[i] = _mm256_setzero_ps();
            q[i] = A.qp(aidx[t0 + i]);
            dq[i] = A.dp(aidx[t0 + i]);
            sq[i] = A.sp(aidx[t0 + i]);
            s16[i] = A.sqp(aidx[t0 + i]);
        }
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < 4; ++c) {
                __m256i u[8];
                for (int j = 0; j < 8; ++j) u[j] = _mm256_loadu_si256((const __m256i*) (ur[j] + 128 * b + 32 * c));
                const __m256 da0 = _mm256_loadu_ps(da + 8 * (8 * b + 2 * c)), da1 = _mm256_loadu_ps(da + 8 * (8 * b + 2 * c + 1));
                const __m256 ma0 = _mm256_loadu_ps(ma + 8 * (8 * b + 6)), ma1 = _mm256_loadu_ps(ma + 8 * (8 * b + 7));
                for (int i = 0; i < tn; ++i) {
                    const __m256i qv = _mm256_loadu_si256((const __m256i*) (q[i] + 128 * b + 32 * c));
                    __m256i v[8];
                    for (int j = 0; j < 8; ++j) v[j] = _mm256_madd_epi16(_mm256_maddubs_epi16(u[j], qv), ones);
                    // v[j]: dwords 0..3 = the first 16 values' products, 4..7 the second 16's
                    const __m256i x = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
                    const __m256i y = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
                    const __m256 c0 = _mm256_cvtepi32_ps(_mm256_permute2x128_si256(x, y, 0x20));   // rows 0..7, group 0
                    const __m256 c1 = _mm256_cvtepi32_ps(_mm256_permute2x128_si256(x, y, 0x31));   // rows 0..7, group 1
                    __m256 tmp = sw1 ? mul_add(nofma, c1, da1, _mm256_mul_ps(c0, da0)) : mul_add(nofma, c0, da0, _mm256_mul_ps(c1, da1));
                    if (c == 3) {
                        const __m256 m0 = _mm256_set1_ps((float) s16[i][8 * b + 6]), m1 = _mm256_set1_ps((float) s16[i][8 * b + 7]);
                        const __m256 t2 = sw2 ? mul_add(nofma, m1, ma1, _mm256_mul_ps(m0, ma0)) : mul_add(nofma, m0, ma0, _mm256_mul_ps(m1, ma1));
                        tmp = _mm256_sub_ps(tmp, t2);
                    }
                    acc[i] = mul_add(nofma, tmp, _mm256_broadcast_ss(dq[i] + 2 * b + (c >> 1)), acc[i]);
                }
            }
            for (int g = 0; g < 6; ++g) {
                const __m256 mg = _mm256_loadu_ps(ma + 8 * (8 * b + g));
                for (int i = 0; i < tn; ++i) {
                    const __m256 sb = _mm256_broadcast_ss(sq[i] + 6 * b + g);
                    acc[i] = (nofma || nof3) ? _mm256_sub_ps(acc[i], _mm256_mul_ps(mg, sb)) : _mm256_fnmadd_ps(mg, sb, acc[i]);
                }
            }
        }
        for (int i = 0; i < tn; ++i) _mm256_storeu_ps(out[t0 + i], acc[i]);
    }
}

/// Rows [r0, r1) (both multiples of 8) of a `type` matrix with K-value rows at `w` against nt activations:
/// out[t][r] for r in [r0, r1).  `out[t]` points at row 0 of activation t's output.  `A` / `AK`: the activation rows
/// in the layout of act_kind(type).  `var`: bit 0 = no fused multiply-add; the others are group8k's.
DS41_CPUX_TARGET inline void rows(int type, const uint8_t* w, int K, int r0, int r1, const Acts& A, const ActsK& AK,
                                  const int32_t* aidx, int nt, float* const* out, int var, Scratch& sc) {
    const size_t rb = row_bytes(type, K);
    float* o[64];
    for (int r = r0; r < r1; r += 8) {
        for (int t = 0; t < nt; ++t) o[t] = out[t] + r;
        if (type == kTypeQ2K) {
            for (int i = 0; i < 8; ++i)
                decode_row_q2k(w + (size_t) (r + i) * rb, K, sc.row(i), sc.da.data() + i, sc.ma.data() + i, 8);
            group8k(sc, K, AK, aidx, nt, o, var);
        } else {
            for (int i = 0; i < 8; ++i) decode_row(type, w + (size_t) (r + i) * rb, K, sc.row(i), sc.da.data() + i, 8);
            group8(sc, K, A, aidx, nt, o, (var & 1) == 0);
        }
    }
}
constexpr int kMaxAct = 64;   ///< activations one rows() call takes (the `o` array)

/// The scalar transcription of the same sums (tests; a CPU without AVX2 is refused by cpu_ok).
inline float row_ref(int type, const uint8_t* wrow, int K, const Acts& A, int a, bool fma, float (*fma_fn)(float, float, float)) {
    std::vector<uint8_t> u((size_t) K + 64);
    std::vector<float> da((size_t) K / 32);
    decode_row(type, wrow, K, u.data(), da.data(), 1);
    float sum = 0.0f;
    const int8_t* q = A.qp(a);
    for (int c = 0; c < K / 32; ++c) {
        int32_t dot = 0;
        for (int i = 0; i < 32; ++i) dot += ((int32_t) u[(size_t) 32 * c + i] - 1) * (int32_t) q[32 * c + i];
        volatile float tm = (float) dot * da[(size_t) c];
        if (fma) sum = fma_fn(tm, A.dbp(a)[c], sum);
        else { volatile float p = tm * A.dbp(a)[c]; sum = sum + p; }
    }
    return sum;
}

#else   // not x86-64

inline bool cpu_ok() { return false; }

#endif

/// A fixed set of threads that runs fn(0) .. fn(n - 1); start() returns at once, wait() when every task is done.
class Pool {
public:
    explicit Pool(int nth) {
        for (int k = 0; k < std::max(1, nth); ++k) th_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_.wait(lk, [&] { return busy_ == 0; });
            stop_ = true;
        }
        work_.notify_all();
        for (auto& t : th_) t.join();
    }
    int threads() const { return (int) th_.size(); }
    void start(int n, std::function<void(int)> fn) {
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk, [&] { return busy_ == 0; });
        fn_ = std::move(fn);
        n_ = n;
        next_.store(0, std::memory_order_relaxed);
        busy_ = (int) th_.size();
        ++gen_;
        lk.unlock();
        work_.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk, [&] { return busy_ == 0; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu_);
        return busy_ == 0;
    }

private:
    void loop() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            work_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            lk.unlock();
            for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n_;) fn_(i);
            lk.lock();
            if (--busy_ == 0) done_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable work_, done_;
    std::function<void(int)> fn_;
    int n_ = 0, busy_ = 0;
    std::atomic<int> next_{0};
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace strata::v41::cpux
