$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
. (Join-Path $PSScriptRoot 'drawing_helpers.ps1')
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$assetDir=Join-Path $root 'kien_truc_dung_mau_paper';[IO.Directory]::CreateDirectory($assetDir)|Out-Null
$sourcePath=Join-Path $root 'demo/vendor/semseg/figure/pspnet.png'
$sourceImage=[Drawing.Image]::FromFile($sourcePath)
$sourceData=[Convert]::ToBase64String([IO.File]::ReadAllBytes($sourcePath))
function X($s){[Security.SecurityElement]::Escape([string]$s)}
function Seg($x1,$y1,$x2,$y2,$color='#222222',$dash=$false){
 $p=[Drawing.Pen]::new((Color $color),1.15);if($dash){$p.DashStyle='Dot'};$g.DrawLine($p,[float]$x1,[float]$y1,[float]$x2,[float]$y2);$p.Dispose()
 $extra=if($dash){' stroke-dasharray="2 3"'}else{''};[void]$svg.AppendLine("<line x1='$x1' y1='$y1' x2='$x2' y2='$y2' stroke='$color' stroke-width='1.15'$extra/>")
}
function Arr($x1,$y1,$x2,$y2,$color='#222222'){
 Seg $x1 $y1 $x2 $y2 $color
 $dx=$x2-$x1;$dy=$y2-$y1;$len=[Math]::Sqrt($dx*$dx+$dy*$dy);if($len -eq 0){return};$ux=$dx/$len;$uy=$dy/$len
 $p1=[Drawing.PointF]::new($x2,$y2);$p2=[Drawing.PointF]::new(($x2-7*$ux+2.7*$uy),($y2-7*$uy-2.7*$ux));$p3=[Drawing.PointF]::new(($x2-7*$ux-2.7*$uy),($y2-7*$uy+2.7*$ux));$br=[Drawing.SolidBrush]::new((Color $color));$g.FillPolygon($br,[Drawing.PointF[]]@($p1,$p2,$p3));$br.Dispose()
 [void]$svg.AppendLine("<polygon points='$($p1.X),$($p1.Y) $($p2.X),$($p2.Y) $($p3.X),$($p3.Y)' fill='$color'/>")
}
function Label($s,$x,$y,$size=20,$color='#111111',$anchor='middle'){
 $f=[Drawing.Font]::new('Times New Roman',$size,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel);$br=[Drawing.SolidBrush]::new((Color $color));$fmt=[Drawing.StringFormat]::new();$fmt.Alignment=if($anchor -eq 'middle'){'Center'}elseif($anchor -eq 'end'){'Far'}else{'Near'};$fmt.LineAlignment='Center'
 $g.DrawString($s,$f,$br,[Drawing.PointF]::new($x,$y),$fmt);$fmt.Dispose();$br.Dispose();$f.Dispose();$es=X $s
 [void]$svg.AppendLine("<text x='$x' y='$y' text-anchor='$anchor' dominant-baseline='central' font-family='Times New Roman,serif' font-size='$size' fill='$color'>$es</text>")
}
function OutlineBox($s,$x,$y,$w,$h,$size=20){
 $br=[Drawing.SolidBrush]::new([Drawing.Color]::White);$g.FillRectangle($br,$x,$y,$w,$h);$br.Dispose();[void]$svg.AppendLine("<rect x='$x' y='$y' width='$w' height='$h' fill='white' stroke='#222' stroke-width='1.15'/>")
 $p=[Drawing.Pen]::new([Drawing.Color]::FromArgb(40,40,40),1.15);$g.DrawRectangle($p,[float]$x,[float]$y,[float]$w,[float]$h);$p.Dispose();Label $s ($x+$w/2) ($y+$h/2) $size
}
function Tensor($x,$y,$w,$h,$dx,$dy,$color,$grid=0){
 # Front rectangle and the two visible oblique faces use the paper's wireframe grammar.
 $points=[Drawing.PointF[]]@([Drawing.PointF]::new($x,$y),[Drawing.PointF]::new(($x+$w),$y),[Drawing.PointF]::new(($x+$w+$dx),($y+$dy)),[Drawing.PointF]::new(($x+$w+$dx),($y+$h+$dy)),[Drawing.PointF]::new(($x+$dx),($y+$h+$dy)),[Drawing.PointF]::new($x,($y+$h)))
 $br=[Drawing.SolidBrush]::new([Drawing.Color]::White);$g.FillPolygon($br,$points);$br.Dispose();$pts=($points|ForEach-Object{"$($_.X),$($_.Y)"}) -join ' ';[void]$svg.AppendLine("<polygon points='$pts' fill='white'/>")
 Seg $x $y ($x+$w) $y $color;Seg $x $y ($x+$dx) ($y+$dy) $color;Seg ($x+$w) $y ($x+$w+$dx) ($y+$dy) $color;Seg ($x+$dx) ($y+$dy) ($x+$w+$dx) ($y+$dy) $color
 Seg $x $y $x ($y+$h) $color;Seg $x ($y+$h) ($x+$dx) ($y+$h+$dy) $color;Seg ($x+$dx) ($y+$dy) ($x+$dx) ($y+$h+$dy) $color;Seg ($x+$w+$dx) ($y+$dy) ($x+$w+$dx) ($y+$h+$dy) $color;Seg ($x+$dx) ($y+$h+$dy) ($x+$w+$dx) ($y+$h+$dy) $color
 if($grid -gt 1){for($j=1;$j -lt $grid;$j++){
  $xx=$x+$w*$j/$grid;Seg $xx $y ($xx+$dx) ($y+$dy) $color
  $yy=$y+$h*$j/$grid;Seg $x $yy ($x+$dx) ($yy+$dy) $color;Seg ($x+$dx) ($yy+$dy) ($x+$w+$dx) ($yy+$dy) $color
  Seg ($x+$dx+$w*$j/$grid) ($y+$dy) ($x+$dx+$w*$j/$grid) ($y+$h+$dy) $color
 }}
}
function CirclePlus($x,$y){$p=[Drawing.Pen]::new([Drawing.Color]::FromArgb(40,40,40),1.1);$br=[Drawing.SolidBrush]::new([Drawing.Color]::White);$g.FillEllipse($br,($x-9),($y-9),18,18);$g.DrawEllipse($p,($x-9),($y-9),18,18);$p.Dispose();$br.Dispose();[void]$svg.AppendLine("<circle cx='$x' cy='$y' r='9' fill='white' stroke='#222' stroke-width='1.1'/>");Label '+' $x $y 18}
function Photo($x,$y,$w,$h,$sx,$sy,$sw,$sh){
 $g.DrawImage($sourceImage,[Drawing.RectangleF]::new($x,$y,$w,$h),[Drawing.RectangleF]::new($sx,$sy,$sw,$sh),[Drawing.GraphicsUnit]::Pixel)
 [void]$svg.AppendLine("<svg x='$x' y='$y' width='$w' height='$h' viewBox='$sx $sy $sw $sh' preserveAspectRatio='none'><image width='$($sourceImage.Width)' height='$($sourceImage.Height)' href='data:image/png;base64,$sourceData'/></svg>")
}
function Paper($kind,$k){
 $H=if($kind -eq 'aspp'){660}else{465};$W=1700;$bitmap=[Drawing.Bitmap]::new(($W*3),($H*3));$g=[Drawing.Graphics]::FromImage($bitmap);$g.Clear([Drawing.Color]::White);$g.ScaleTransform(3,3);$g.SmoothingMode='AntiAlias';$g.TextRenderingHint='AntiAliasGridFit'
 $svg=[Text.StringBuilder]::new();[void]$svg.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$($W*3)' height='$($H*3)' viewBox='0 0 $W $H'><rect width='$W' height='$H' fill='white'/>")
 Photo 20 155 230 154 0 350 890 590
 Arr 250 230 265 230;OutlineBox 'CNN' 268 215 48 30 21;Arr 316 230 350 230
 Tensor 350 130 40 170 58 38 '#222222' 0
 Arr 450 230 475 230;OutlineBox 'POOL' 478 215 58 30 20;Arr 536 230 580 230
 # PPM dotted boundary and grouping brackets.
 $bottom=if($kind -eq 'aspp'){608}else{405}
 Seg 465 35 1295 35 '#555555' $true;Seg 465 35 465 $bottom '#555555' $true;Seg 465 $bottom 1295 $bottom '#555555' $true;Seg 1295 35 1295 $bottom '#555555' $true
 Seg 590 75 580 78;Seg 580 78 580 380;Seg 580 380 590 383
 $ys=@(100,173,253,345);$cols=@('#FF2638','#FF8518','#00A9E0','#00B466');$ns=@(1,2,3,[int]$k)
 for($i=0;$i -lt 4;$i++){
  $cy=$ys[$i];$n=$ns[$i];$tw=@(36,43,50,62)[$i];$th=@(10,32,45,70)[$i];$dep=@(7,11,15,21)[$i];$dy=@(4,7,10,14)[$i];$top=$cy-($th+$dy)/2
  Tensor 630 $top $tw $th $dep $dy $cols[$i] $n
  Label "$n×$n" 618 $cy 15 '#111111' 'end'
  Arr (630+$tw+$dep+4) $cy 750 $cy;OutlineBox 'CONV' 753 ($cy-13) 62 26 19;Arr 815 $cy 850 $cy
  $sw=@(10,13,17,21)[$i];$sh=@(10,22,31,42)[$i];$sd=@(4,5,7,10)[$i]
  Tensor 854 ($cy-($sh+$sd)/2) $sw $sh $sd $sd $cols[$i] $n
  $sourceX=854+$sw+$sd
  if($kind -eq 'link'){
   Arr ($sourceX+4) $cy 1022 $cy;CirclePlus 1032 $cy;Arr 1042 $cy 1070 $cy
   Label "z$($i+1)" 880 ($cy+$sh/2+17) 13
  }else{Arr ($sourceX+4) $cy 1070 $cy}
 }
 if($kind -eq 'link'){
  for($i=0;$i -lt 3;$i++){
   # Both directions start at the ORIGINAL branch features, never at mixed features.
   Arr 906 $ys[$i] 1026 ($ys[$i+1]-7) '#A23479'
   Arr 906 $ys[$i+1] 1026 ($ys[$i]+7) '#A23479'
  }
  Label 'αij = tanh(wij); resize / pool → × αij → +' 845 425 16 '#5E2450'
 }
 Seg 1070 75 1080 78;Seg 1080 78 1080 380;Seg 1080 380 1070 383;Arr 1080 230 1106 230
 OutlineBox '' 1109 177 29 107 19
 $state=$g.Save();$g.TranslateTransform(1123.5,230.5);$g.RotateTransform(90);$font=[Drawing.Font]::new('Times New Roman',18,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel);$sf=[Drawing.StringFormat]::new();$sf.Alignment='Center';$sf.LineAlignment='Center';$g.DrawString('UPSAMPLE',$font,[Drawing.Brushes]::Black,[Drawing.PointF]::new(0,0),$sf);$sf.Dispose();$font.Dispose();$g.Restore($state)
 [void]$svg.AppendLine("<text transform='translate(1123.5 230.5) rotate(90)' text-anchor='middle' dominant-baseline='central' font-family='Times New Roman,serif' font-size='18'>UPSAMPLE</text>")
 Arr 1138 230 1175 230
 # Layered tensor and long backbone skip use the same geometry as the reference.
 Tensor 1182 130 43 170 58 38 '#222222' 0
 for($i=0;$i -lt 4;$i++){$xx=1233+$i*8;Seg ($xx-58) 130 $xx 168 $cols[$i];Seg $xx 168 $xx 338 $cols[$i];Seg ($xx-14) 338 $xx 338 $cols[$i]}
 if($kind -eq 'aspp'){for($j=0;$j -lt 3;$j++){$xx=1265+$j*6;Seg ($xx-58) 130 $xx 168 '#9254AE';Seg $xx 168 $xx 338 '#9254AE';Seg ($xx-12) 338 $xx 338 '#9254AE'}}
 Seg 390 130 390 20;Seg 390 20 1215 20;Arr 1215 20 1215 151
 Label 'CONCAT' 1230 365 20
 Arr 1285 230 1320 230;OutlineBox 'CONV' 1322 215 62 30 20;Arr 1384 230 1410 230
 Photo 1412 155 230 154 4215 365 889 595
 if($kind -eq 'aspp'){
  Seg 450 300 455 300;Seg 455 300 455 433;Seg 455 433 1070 433
  $centers=@(640,830,1020)
  for($j=0;$j -lt 3;$j++){
   $cx=$centers[$j];Arr $cx 433 $cx 470;OutlineBox "3×3, d=$(@(12,24,36)[$j])" ($cx-67) 473 134 28 17
   Label 'BN + ReLU' $cx 515 13;Arr $cx 526 $cx 543
   Tensor ($cx-12) 545 19 21 7 5 '#9254AE' 0;Label '256' ($cx+34) 558 13
   Seg $cx 573 $cx 588
  }
  Seg 640 588 1302 588;Seg 1302 588 1302 340;Arr 1302 340 1284 331
  Label '4096 + 3×256 = 4864' 1230 391 16
 }
 $labelY=$H-21
 Label '(a) Input Image' 135 $labelY 24;Label '(b) Feature Map' 393 $labelY 24
 Label $(if($kind -eq 'aspp'){'(c) Pyramid Pooling Module + Atrous'}else{'(c) Pyramid Pooling Module'}) 867 $labelY 24
 Label '(d) Final Prediction' 1527 $labelY 24
 [void]$svg.AppendLine('</svg>')
 $suffix=if($kind -eq 'base'){''}else{'_'+$kind};$name="b$k$suffix"
 $bitmap.Save((Join-Path $assetDir "$name.png"),[Drawing.Imaging.ImageFormat]::Png);[IO.File]::WriteAllText((Join-Path $assetDir "$name.svg"),$svg.ToString(),[Text.UTF8Encoding]::new($false));$g.Dispose();$bitmap.Dispose()
}
foreach($kind in @('base','link','aspp')){foreach($k in @('6','7')){Paper $kind $k}}
$sourceImage.Dispose()
Write-Output "Created six PNG and six editable SVG diagrams in $assetDir"
