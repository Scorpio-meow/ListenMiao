param(
    [switch]$NoBrowser,
    [switch]$Tray
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Show-Box([string]$message, [string]$icon) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($message, "ListenMiao", "OK", $icon) | Out-Null
}

function Show-Warning([string]$message) {
    if ($Tray) {
        Show-Box $message "Warning"
        return
    }
    Write-Warning $message
}

function Wait-BeforeClose([string]$message) {
    if ($Tray) {
        Show-Box $message "Error"
        return
    }
    Write-Host ""
    Write-Host $message -ForegroundColor Red
    Read-Host "按 Enter 關閉視窗" | Out-Null
}

trap {
    Wait-BeforeClose "啟動失敗：$_"
    exit 1
}

if (-not (Test-Path .env)) {
    throw "找不到 .env，請先執行 scripts\install.ps1"
}

uv run --no-sync python -m app.envfile .env.example .env
if ($LASTEXITCODE -ne 0) {
    Show-Warning "無法檢查 .env 是否缺少設定，照原本的 .env 繼續啟動"
}

$settings = @{}
foreach ($line in Get-Content .env -Encoding UTF8) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        $settings[$Matches[1]] = $Matches[2]
    }
}
$url = uv run --no-sync --env-file .env python -c "from app.cli import load; print(load().service_url(None))"
if ($LASTEXITCODE -ne 0) {
    throw "無法從 .env 的 ASR_HOST、ASR_PORT 組出服務網址"
}

function Test-Service {
    try {
        Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if ($Tray) {
    $instance = New-Object System.Threading.Mutex($false, "ListenMiao-$($url -replace '[^A-Za-z0-9]', '-')")
    $owned = $false
    while (-not $owned) {
        if (Test-Service) {
            Start-Process $url
            exit 0
        }
        try {
            $owned = $instance.WaitOne(1000)
        } catch [System.Threading.AbandonedMutexException] {
            $owned = $true
        }
    }
}

if (-not $NoBrowser) {
    Start-Job -ArgumentList $url -ScriptBlock {
        param($url)
        while ($true) {
            try {
                Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
                Start-Process $url
                return
            } catch {
                Start-Sleep -Seconds 1
            }
        }
    } | Out-Null
}

$request = Join-Path $root ".update\request.json"
$logs = Join-Path $settings["ASR_DATA_DIR"] "logs"
$console = Join-Path $logs "console.log"
while ($true) {
    Write-Host "ListenMiao：$url（關閉這個視窗即停止服務）"
    if ($Tray) {
        New-Item -ItemType Directory -Force $logs | Out-Null
        $env:PYTHONIOENCODING = "utf-8"
        cmd /c "uv run --no-sync --env-file .env python -m app --updatable --tray > `"$console`" 2>&1"
    } else {
        uv run --no-sync --env-file .env python -m app --updatable
    }
    $code = $LASTEXITCODE
    if (-not (Test-Path $request)) {
        break
    }
    Write-Host ""
    Write-Host "下載並套用新版…"
    uv run --no-sync python -m app.update --root . apply --yes
    if ($LASTEXITCODE -eq 0) {
        $group = uv run --no-sync python -m app.update --root . group
        if ($LASTEXITCODE -eq 0) {
            uv sync --frozen --no-default-groups --group $group
        }
        if ($LASTEXITCODE -ne 0) {
            Show-Warning "安裝新版的套件失敗，還原成更新前的檔案"
            uv run --no-sync python -m app.update --root . rollback
            $group = uv run --no-sync python -m app.update --root . group
            uv sync --frozen --no-default-groups --group $group
        }
    } else {
        Show-Warning "更新沒有完成，繼續用原本的版本"
    }
    uv run --no-sync python -m app.envfile .env.example .env
    Write-Host ""
}
if ($code -ne 0 -and $Tray) {
    $last = (Get-Content $console -Encoding UTF8 | Where-Object { $_.Trim() } | Select-Object -Last 5) -join "`n"
    Wait-BeforeClose "服務異常結束（結束代碼 $code）。最後的輸出：`n`n$last`n`n完整內容在 $console，其他日誌在 $logs。"
    exit $code
}
if ($code -ne 0) {
    Wait-BeforeClose "服務異常結束（結束代碼 $code）。原因見上方訊息，或 $logs 裡的日誌。"
    exit $code
}
