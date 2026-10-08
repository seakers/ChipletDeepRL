# Keep the system and display awake while any runner is alive (Modern Standby suspends them when the screen turns off)
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000003") | Out-Null
while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like '*runner.py*' }).Count -gt 0) { Start-Sleep 60 }
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
"keepawake released $(Get-Date -Format HH:mm)"
