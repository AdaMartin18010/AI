Param(
    [string[]]$Roots = @("Analysis","Matter"),
    [string]$Report = "../link_report.txt"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 运行链接检查
$python = "python"
$cmd = "$python link_check.py $($Roots -join ' ')"

Write-Host "Running: $cmd"
$proc = Start-Process -FilePath $python -ArgumentList (@("link_check.py") + $Roots) -NoNewWindow -RedirectStandardOutput $Report -RedirectStandardError "$Report.err" -PassThru
$proc.WaitForExit()

# 输出报告位置
Write-Host "Report written to: $(Resolve-Path $Report)"

# 简要统计
$content = Get-Content $Report -Raw
$broken = ($content | Select-String -Pattern "^BROKEN ").Count
if ($broken -eq $null) { $broken = 0 }
Write-Host "BROKEN files: $broken" 