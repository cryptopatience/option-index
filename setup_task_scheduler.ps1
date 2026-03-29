# setup_task_scheduler.ps1
# 관리자 권한으로 실행: PowerShell을 우클릭 → "관리자로 실행" 후 아래 명령
#   cd "c:\Users\user\옵션 저점매수전략"
#   .\setup_task_scheduler.ps1

$TaskName   = "옵션전략_Discord알림_08시"
$ScriptPath = "c:\Users\user\옵션 저점매수전략\discord_notify.py"

# Python 경로 자동 감지
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Error "Python을 찾을 수 없습니다. PATH에 Python이 등록되어 있는지 확인하세요."
    exit 1
}
Write-Host "Python 경로: $PythonExe"

# 기존 태스크 제거 (재등록 시)
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# 트리거: 매일 08:00
$Trigger = New-ScheduledTaskTrigger -Daily -At "08:00"

# 액션: python discord_notify.py
$Action  = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory "c:\Users\user\옵션 저점매수전략"

# 설정: 배터리 상태 무관, 네트워크 연결 후 실행
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# 현재 로그인 사용자로 등록
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Trigger   $Trigger `
    -Action    $Action `
    -Settings  $Settings `
    -Principal $Principal `
    -Force | Out-Null

Write-Host "✅ Task Scheduler 등록 완료: '$TaskName'"
Write-Host "   매일 오전 08:00 에 discord_notify.py 가 자동 실행됩니다."
Write-Host ""
Write-Host "지금 바로 테스트 실행:"
Write-Host "   python `"$ScriptPath`""
