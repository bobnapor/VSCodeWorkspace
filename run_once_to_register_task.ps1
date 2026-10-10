# run_once_to_register_task.ps1
# Run this script ONCE in PowerShell (as Administrator) to register the bot
# as a Windows Scheduled Task that starts automatically at login.
#
# To run:   Right-click → "Run with PowerShell"
# To remove: Open Task Scheduler → find "CryptoSignalBot" → Delete

$pythonw = "C:\Users\Bobby\AppData\Local\Programs\Python\Python38-32\pythonw.exe"
$script  = "e:\PythonProjects\VSCodeWorkspace\crypto_main.py"
$workdir = "e:\PythonProjects\VSCodeWorkspace"

$action  = New-ScheduledTaskAction -Execute $pythonw -Argument $script -WorkingDirectory $workdir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartOnIdle $false

Register-ScheduledTask `
    -TaskName   "CryptoSignalBot" `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Description "Runs the crypto signal bot silently at Windows login" `
    -Force

Write-Host "Task registered. The bot will start automatically next time you log in."
Write-Host "To start it now without logging out, run:"
Write-Host "  Start-ScheduledTask -TaskName 'CryptoSignalBot'"
