# Start a command as a detached process (WMI) that keeps running after the shell or terminal that started it closes.
#   powershell -File windows-local\start_detached.ps1 -cmd 'cmd /c "cd /d D:\src\Xolotl && .venv\Scripts\python.exe -I -X utf8 tools\v41\serve\serve_ds41.py --engine strata --config D:\models\xolotl-dsv41-xq-v4-fast-win.json --port 8150 > D:\models\xolotl-dsv41-serve.log 2>&1"'
# Stop it later with: Get-Process ds41_generate, strata | Stop-Process   (and the python server process)
param([string]$cmd)
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
