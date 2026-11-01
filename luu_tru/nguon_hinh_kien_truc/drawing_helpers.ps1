Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root=Split-Path $PSScriptRoot -Parent
$out=$PSScriptRoot
function Color($s){[Drawing.ColorTranslator]::FromHtml($s)}
function Text($g,$s,$x,$y,$w,$h,$size=27,$color='#172B4D'){
 $f=[Drawing.Font]::new('Arial',$size,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel)
 $b=[Drawing.SolidBrush]::new((Color $color));$fmt=[Drawing.StringFormat]::new();$fmt.Alignment='Center';$fmt.LineAlignment='Center'
 $g.DrawString($s,$f,$b,[Drawing.RectangleF]::new($x,$y,$w,$h),$fmt);$f.Dispose();$b.Dispose();$fmt.Dispose()
}
function Box($g,$s,$x,$y,$w,$h,$c='#EDF4FB',$size=27){
 $b=[Drawing.SolidBrush]::new((Color $c));$p=[Drawing.Pen]::new((Color '#AAB9CE'),2)
 $g.FillRectangle($b,$x,$y,$w,$h);$g.DrawRectangle($p,$x,$y,$w,$h);Text $g $s $x $y $w $h $size;$b.Dispose();$p.Dispose()
}
function Arrow($g,$x1,$y1,$x2,$y2,$c='#536780'){
 $p=[Drawing.Pen]::new((Color $c),3);$p.CustomEndCap=[Drawing.Drawing2D.AdjustableArrowCap]::new(5,6);$g.DrawLine($p,$x1,$y1,$x2,$y2);$p.Dispose()
}
function ReadXml($p){$x=[xml]::new();$x.PreserveWhitespace=$true;$x.Load($p);return ,$x}
function NS($x){$n=[Xml.XmlNamespaceManager]::new($x.NameTable);foreach($v in @(@('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main'),@('a','http://schemas.openxmlformats.org/drawingml/2006/main'),@('p','http://schemas.openxmlformats.org/presentationml/2006/main'),@('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships'),@('wp','http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'))){$n.AddNamespace($v[0],$v[1])};return ,$n}