param(
    [Parameter(Mandatory=$true)][string]$ScriptPath,
    [Parameter(Mandatory=$true)][string]$StatusFile,
    [string]$WorkDir = ".",
    [int]$IntervalSec = 15,
    [string[]]$ArgsList = @()
)
New-Item -ItemType Directory -Force -Path (Split-Path $StatusFile) | Out-Null
$outFile = "$StatusFile.out"
$errFile = "$StatusFile.err"
$ps = { param($sp, $af, $o, $e)
    (Get-Item $sp).Directory.Create() | Out-Null
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $af -WorkingDirectory (Split-Path $sp) -RedirectStandardOutput $o -RedirectStandardError $e -PassThru -WindowStyle Hidden
    $p.Id
}
$pidStr = & $ps $ScriptPath $ArgsList $outFile $errFile
"STARTED pid=$pidStr" | Out-File $StatusFile -Encoding utf8
while (Get-Process -Id $pidStr -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $IntervalSec
    $lines = if (Test-Path $outFile) { (Get-Content $outFile -ErrorAction SilentlyContinue) } else { @() }
    $tail = if ($lines.Count -gt 2) { ($lines | Select-Object -Last 1) } else { ($lines -join '') }
    "$(Get-Date -Format o) tail=[$tail]"  | Out-File $StatusFile -Append -Encoding utf8
}
"COMPLETED rc=$(Get-Content $outFile -ErrorAction SilentlyContinue | Select-Object -Last 3 | Out-String)" | Out-File $StatusFile -Append -Encoding utf8