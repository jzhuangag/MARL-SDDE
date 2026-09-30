$ErrorActionPreference = 'Stop'
$ManuscriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FigureScript = Join-Path $ManuscriptRoot 'make_multienvironment_return_figure.py'
$Python = 'D:\anaconda\envs\ust2\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
$Python = 'python'
}
& $Python $FigureScript
Push-Location $ManuscriptRoot
try {
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}
finally {
Pop-Location
}
