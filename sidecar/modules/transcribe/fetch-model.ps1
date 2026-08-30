<#
.SYNOPSIS
Downloads the whisper base files transcribe compiles in.

.DESCRIPTION
The weights are 290 MB, so they are not in git. This puts them in model/,
beside this script, where the module's build script looks for them. Every file
is checked against a pinned sha256; a file already present and matching is left
alone, so running this twice downloads nothing the second time.

Build the module again afterwards - the weights are compiled into the wasm, so
a build made before the download has none.
#>

[CmdletBinding()]
param(
    # Download and rewrite even files that are already present and correct.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# openai/whisper-base, at the commit these hashes were taken from. The
# multilingual model, not base.en: transcribe takes a language.
$Revision = 'e37978b90ca9030d5170a5c07aadb050351a65bb'
$BaseUrl = "https://huggingface.co/openai/whisper-base/resolve/$Revision"

$Files = @(
    @{ Name = 'model.safetensors'; Sha256 = '07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b' }
    @{ Name = 'tokenizer.json';    Sha256 = '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566' }
    @{ Name = 'config.json';       Sha256 = 'a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1' }
)

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
    Invoke-WebRequest -Uri "$BaseUrl/$($file.Name)" -OutFile $temp

    $got = (Get-FileHash -Path $temp -Algorithm SHA256).Hash
    if ($got -ine $file.Sha256) {
        Remove-Item $temp -Force
        throw "$($file.Name): downloaded file has sha256 $got, expected $($file.Sha256)"
    }
    Move-Item -Path $temp -Destination $path -Force
    Write-Host "$($file.Name): ok"
}

Write-Host ''
Write-Host "The whisper files are in $ModelDir."
Write-Host 'Build the module again to compile them in:'
Write-Host '  cargo build --release --target wasm32-wasip2 -p transcribe'
