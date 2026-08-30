<#
.SYNOPSIS
Downloads the YOLOv8n-seg ONNX graph the segment module runs.

.DESCRIPTION
The graph is 14 MB, so it is not in git. This puts it in model/, beside this
script, and leaves a copy called segment.onnx next to the built wasm, which is
where the compiler emits its -nn binding from. The module does not compile it
in: the host loads it and the module reaches it by name, so nothing needs
rebuilding afterwards. Bind it when you run:

  ffrwd-wasm -nn segment=modules/segment/model/yolov8n-seg.onnx -nn-runtime <dir> ...

The file is checked against a pinned sha256; one already present and matching
is left alone, so running this twice downloads nothing the second time.
#>

[CmdletBinding()]
param(
    # Download and rewrite even a file that is already present and correct.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# mobilint/YOLOv8n-seg, at the commit this hash was taken from: the fp32 ONNX
# export of Ultralytics' COCO-trained yolov8n-seg, which the repository keeps
# beside the quantized builds it is actually there for. 640x640 in, [1, 116,
# 8400] and [1, 32, 160, 160] out.
$Revision = '7b775a2526a57688597b2abe50dae6ec6567d4cf'
$BaseUrl = "https://huggingface.co/mobilint/YOLOv8n-seg/resolve/$Revision"

$Files = @(
    @{ Name = 'yolov8n-seg.onnx'; Remote = 'yolov8n-seg.onnx'; Sha256 = 'cb1e689c548b3fa019691c9c1762f38a76276981ce75d31f6c5aae396dbff78b' }
)

# The name the host binds, beside the wasm the compiler names.
$BesideWasm = Join-Path $PSScriptRoot '..\target\wasm32-wasip2\release'
$BoundAs = 'segment.onnx'

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
Write-Host "The segment graph is in $ModelDir."
Write-Host 'Bind it by name when you run the sidecar:'
Write-Host "  -nn segment=$(Join-Path $ModelDir $Files[0].Name)"
