# Native GPU profiling, which is one of the two things going native was
# actually worth (see NOTES.md).
#
# Under WSL2 nsys traces the CUDA API fine but the capture contains no GPU
# kernel rows at all, so the kernels had to time themselves with CUDA events
# and those events were the only source of truth. Natively there is a real
# timeline to check them against.
#
#   .\src\profile.ps1                      # nsys capture + kernel/memop summary
#   .\src\profile.ps1 -Target ..\build\windows\bin\unified_engine.exe -TargetArgs '--bench-present'
#   .\src\profile.ps1 -Ncu tiled_raster_kernel    # per-kernel counters (needs admin)
#
# Run from the repo root. The profiled binary is launched from src/ because
# the models resolve as ../obj/... from there.

[CmdletBinding()]
param(
    # What to profile. Defaults to the headless stage-breakdown tool, which is
    # the right target: bounded, no window, and nothing but the pipeline.
    [string]$Target = "..\build\windows\bin\profile_frame.exe",
    [string]$TargetArgs = "",

    # Kernel name (substring) to collect Nsight Compute counters for. When
    # empty, only the nsys timeline is captured.
    [string]$Ncu = "",

    [int]$NcuLaunches = 3,
    [string]$OutDir = "$env:TEMP\tinyrenderer-profile"
)

$ErrorActionPreference = "Stop"

function Find-Tool([string]$exe) {
    # The toolkit does not always ship these; the standalone Nsight installs
    # do, and are the ones that carry a matching host build for the reports.
    $roots = @(
        "C:\Program Files\NVIDIA Corporation",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    )
    foreach ($r in $roots) {
        if (-not (Test-Path $r)) { continue }
        $hit = Get-ChildItem $r -Recurse -Depth 4 -Filter $exe -ErrorAction SilentlyContinue |
               Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

$nsys = Find-Tool "nsys.exe"
if (-not $nsys) { throw "nsys.exe not found. Install Nsight Systems." }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$rep = Join-Path $OutDir "capture"

if (-not (Test-Path "src")) { throw "run this from the repo root" }

Push-Location src
try {
    Write-Host "`n=== nsys capture: $Target $TargetArgs ===" -ForegroundColor Cyan

    $argList = @(
        "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "-o", $rep, $Target
    )
    if ($TargetArgs) { $argList += $TargetArgs.Split(" ") }

    & $nsys @argList | Select-Object -Last 3

    # cuda_gpu_kern_sum is the one that WSL could never produce. If it comes
    # back empty the capture has no GPU rows and something is wrong with the
    # environment rather than with the code being profiled.
    # nsys writes progress and an "already exported" note to stderr even on
    # success. In Windows PowerShell 5.1, redirecting a native command's stderr
    # wraps each line in an ErrorRecord and trips $ErrorActionPreference, so
    # leave stderr alone and filter stdout only.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    foreach ($report in @("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum")) {
        Write-Host "`n=== $report ===" -ForegroundColor Cyan
        & $nsys stats --report $report --format column --force-export=true "$rep.nsys-rep" |
            Where-Object { $_ -notmatch "^Processing|^Generating|^\s*$|assumed file|Consider using" } |
            Select-Object -First 16
    }

    $ErrorActionPreference = $prevEAP

    if ($Ncu) {
        $ncuExe = Find-Tool "ncu.exe"
        if (-not $ncuExe) { throw "ncu.exe not found. Install Nsight Compute." }

        Write-Host "`n=== ncu: $Ncu ===" -ForegroundColor Cyan

        # ncu reports ERR_NVGPUCTRPERM on stderr, so this one does have to be
        # captured; keep it non-terminating for the same 5.1 reason as above.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $out = & $ncuExe --kernel-name $Ncu --launch-count $NcuLaunches `
                         --section SpeedOfLight --print-summary per-kernel `
                         $Target 2>&1
        $ErrorActionPreference = $prevEAP

        if ($out -match "ERR_NVGPUCTRPERM") {
            Write-Warning @"
Nsight Compute cannot read GPU performance counters as this user.

RmProfilingAdminOnly under
HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak
is already 0 here and that was not enough, so run this from an
ELEVATED shell instead:

  .\src\profile.ps1 -Ncu $Ncu

nsys above needs no elevation and is unaffected.
"@
        } else {
            $out | Select-Object -First 40
        }
    }
}
finally {
    Pop-Location
}

Write-Host "`ncapture written to $rep.nsys-rep" -ForegroundColor DarkGray
Write-Host "open it with the Nsight Systems GUI for the timeline" -ForegroundColor DarkGray
