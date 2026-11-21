@echo off
rem Builds both Xolotl engines natively on Windows (no WSL):
rem   build-win\ds41_generate.exe   DeepSeek-V4.1 (through build-win.bat)
rem   build-cuda12\strata.exe       Qwen3.8-Flash-Next (setup.py's CUDA 12 flags)
rem Needs: Visual Studio 2022 "Desktop development with C++", the CUDA 12.8 toolkit (CUDA_PATH pointing at it), and a
rem .venv-build with cmake + ninja in the repo root:
rem   python -m venv .venv-build && .venv-build\Scripts\pip install cmake ninja
rem Optional: CUDA_ARCH (default 86 = RTX 30xx; 89 = RTX 40xx; 120 = RTX 50xx), JOBS (default 12).
setlocal
cd /d "%~dp0.."
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=86"
if "%JOBS%"=="" set "JOBS=12"
if "%VCVARS%"=="" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" (
  echo build-both: Visual Studio 2022 C++ tools not found. Install "Desktop development with C++" or set VCVARS.
  exit /b 1
)
if not exist ".venv-build\Scripts\cmake.exe" (
  echo build-both: make the build venv first: python -m venv .venv-build ^&^& .venv-build\Scripts\pip install cmake ninja
  exit /b 1
)
echo ==== ds41_generate: configure %TIME%
call build-win.bat configure || exit /b 2
echo ==== ds41_generate: build %TIME%
call build-win.bat ds41_generate || exit /b 3
echo ==== strata: configure %TIME%
call "%VCVARS%" >nul || exit /b 4
set "PATH=%CD%\.venv-build\Scripts;%CUDA_PATH%\bin;%PATH%"
set "ROOT=%CD:\=/%"
cmake -G Ninja -S . -B build-cuda12 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
  "-DCMAKE_CUDA_COMPILER=%CUDA_PATH:\=/%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" ^
  "-DSTRATA_GGML_DIR=%ROOT%/third_party/llama.cpp" -DSTRATA_EXPERIMENTAL_SM60=ON || exit /b 5
echo ==== strata: build %TIME%
cmake --build build-cuda12 --target strata -j %JOBS% || exit /b 6
echo ==== done %TIME%: build-win\ds41_generate.exe and build-cuda12\strata.exe
