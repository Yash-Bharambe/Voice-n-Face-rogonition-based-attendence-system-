$ErrorActionPreference = 'Stop'
# A restricted tool session can inject a deliberately unreachable proxy.
# Do not let a detached server inherit it and lose access to Supabase.
foreach ($proxyName in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')) {
    $proxyValue = [Environment]::GetEnvironmentVariable($proxyName, 'Process')
    if ($proxyValue -and $proxyValue -match '^https?://(127\.0\.0\.1|localhost):9/?$') {
        throw 'This launch environment has a blocked network proxy. Run start_local.ps1 in your normal local terminal or an approved unrestricted launch environment.'
    }
}
$projectDirectory = $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Install the project virtual environment first.'
}
New-Item -ItemType Directory -Path (Join-Path $projectDirectory 'temp') -Force | Out-Null
$services = @(
    @{Name = 'landing'; Directory = 'landing'; Port = 5002; Arguments = @('app.py')},
    @{Name = 'attendance'; Directory = 'app'; Port = 8501; Arguments = @('-m', 'streamlit', 'run', 'app.py', '--server.address=127.0.0.1')}
)
foreach ($service in $services) {
    $existingListener = Get-NetTCPConnection -State Listen -LocalPort $service.Port -ErrorAction SilentlyContinue
    if ($existingListener) {
        Write-Host "$($service.Name): port $($service.Port) is already in use; no duplicate started."
        continue
    }
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $service.Arguments `
        -WorkingDirectory (Join-Path $projectDirectory $service.Directory) -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $projectDirectory "temp/$($service.Name).out.log") `
        -RedirectStandardError (Join-Path $projectDirectory "temp/$($service.Name).err.log") -PassThru
    Set-Content -LiteralPath (Join-Path $projectDirectory "temp/$($service.Name).pid") -Value $process.Id
    Write-Host "$($service.Name) started (PID $($process.Id)): http://localhost:$($service.Port)"
}
