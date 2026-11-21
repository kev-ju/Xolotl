# Start a command as a detached process (WMI) that keeps running after the shell or terminal that started it closes.
#   powershell -File windows-local\start_detached.ps1 -cmd 'cmd /c "cd /d D:\src\Xolotl && .venv\Scripts\python.exe -I -X utf8 tools\v41\serve\serve_ds41.py --engine strata --config D:\models\xolotl-dsv41-xq-v4-fast-win.json --port 8150 > D:\models\xolotl-dsv41-serve.log 2>&1"'
# Stop it later with: Get-Process ds41_generate, strata | Stop-Process   (and the python server process)
# Hidden by default (ShowWindow 0): otherwise Windows 11 opens a blank Windows Terminal window for it, and closing that
# window kills the server. -Show gives it a visible console.
param([string]$cmd, [switch]$Show)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]$(if ($Show) { 1 } else { 0 })}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; ProcessStartupInformation = $si}
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
