# report/rapor.docx -> report/rapor.pdf (Microsoft Word COM ile)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $root "rapor.docx"
$pdf = Join-Path $root "rapor.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docx, $false, $true)
    $doc.Fields.Update() | Out-Null
    $doc.ExportAsFixedFormat([string]$pdf, 17)   # 17 = wdExportFormatPDF
    $doc.Close([ref]0)
} finally {
    $word.Quit()
}
# Sayfa sayısı PDF'ten okunur (Word'ün ComputeStatistics değeri gizli pencerede dışa aktarılan PDF ile uyuşmuyor)
$pages = & python -c "import sys, pymupdf; print(pymupdf.open(sys.argv[1]).page_count)" $pdf
Write-Output "PDF: $pdf ($pages sayfa)"
