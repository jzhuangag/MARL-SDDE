$ErrorActionPreference = 'Stop'
$TspRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $TspRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = 'python'
}
& $Python (Join-Path $TspRoot 'figures\make_figures.py')
Push-Location $TspRoot
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    $PdfToPpm = Get-Command pdftoppm -ErrorAction SilentlyContinue
    if ($null -ne $PdfToPpm) {
        $RenderRoot = Join-Path $TspRoot 'tmp\rendered'
        New-Item -ItemType Directory -Force -Path $RenderRoot | Out-Null
        Get-ChildItem -LiteralPath $RenderRoot -Filter 'page-*.png' -File |
            Remove-Item -Force
        & $PdfToPpm.Source -png -r 130 main.pdf (Join-Path $RenderRoot 'page')
    }
}
finally {
    Pop-Location
}
