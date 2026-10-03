$ErrorActionPreference = 'Stop'
$ManuscriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FigureScripts = @(
    (Join-Path $ManuscriptRoot 'make_controlled_figures.py'),
    (Join-Path $ManuscriptRoot 'make_multienvironment_return_figure.py')
)
$Python = 'D:\anaconda\envs\ust2\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
$Python = 'python'
}
foreach ($FigureScript in $FigureScripts) {
    & $Python $FigureScript
}
Push-Location $ManuscriptRoot
try {
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}
finally {
Pop-Location
}
