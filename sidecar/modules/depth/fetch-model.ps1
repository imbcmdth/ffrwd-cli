<#
.SYNOPSIS
Downloads the Depth Anything V2 (small) ONNX graph the depth module runs.

.DESCRIPTION
The graph is 94 MB, so it is not in git. This puts it in model/, beside this
script, and leaves a copy called depth.onnx next to the built wasm, which is
where the compiler emits its -nn binding from. Unlike transcribe, the module
does not compile it in: the host loads it and the module reaches it by name,
so nothing needs rebuilding afterwards. Bind it when you run:

  ffrwd-wasm -nn depth=modules/depth/model/model.onnx -nn-runtime <dir> ...

The file is checked against a pinned sha256; one already present and matching
is left alone, so running this twice downloads nothing the second time.
#>

[CmdletBinding()]
param(
    # Download and rewrite even a file that is already present and correct.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# onnx-community/depth-anything-v2-small, at the commit this hash was taken
# from. The fp32 export, not one of the quantized ones: the quantized graphs
# lose enough range that a normalized depth map bands visibly.
$Revision = '4472b7362082ad9968fee890ca0f1e5aca36b93d'
$BaseUrl = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/$Revision"

$Files = @(
    @{ Name = 'model.onnx'; Remote = 'onnx/model.onnx'; Sha256 = 'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c' }
)

# The name the host binds, beside the wasm the compiler names.
$BesideWasm = Join-Path $PSScriptRoot '..\target\wasm32-wasip2\release'
$BoundAs = 'depth.onnx'

$ModelDir = Join-Path $PSScriptRoot 'model'
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir | Out-Null
}

foreach ($file in $Files) {
    $path = Join-Path $ModelDir $file.Name

    if ((Test-Path $path) -and -not $Force) {
        $have = (Get-FileHash -Path $path -Algorithm SHA256).Hash
        if ($have -ieq $file.Sha256) {
            Write-Host "$($file.Name): already here"
            continue
        }
        Write-Host "$($file.Name): present but does not match its hash, downloading again"
    }

    Write-Host "$($file.Name): downloading"
    $temp = "$path.part"
    Invoke-WebRequest -Uri "$BaseUrl/$($file.Remote)" -OutFile $temp

    $got = (Get-FileHash -Path $temp -Algorithm SHA256).Hash
    if ($got -ine $file.Sha256) {
        Remove-Item $temp -Force
        throw "$($file.Name): downloaded file has sha256 $got, expected $($file.Sha256)"
    }
    Move-Item -Path $temp -Destination $path -Force
    Write-Host "$($file.Name): ok"
}

if (-not (Test-Path $BesideWasm)) {
    New-Item -ItemType Directory -Path $BesideWasm -Force | Out-Null
}
$copy = Join-Path $BesideWasm $BoundAs
Copy-Item -Path (Join-Path $ModelDir $Files[0].Name) -Destination $copy -Force
Write-Host "$BoundAs`: beside the wasm"

Write-Host ''
Write-Host "The depth graph is in $ModelDir."
Write-Host 'Bind it by name when you run the sidecar:'
Write-Host "  -nn depth=$(Join-Path $ModelDir 'model.onnx')"
