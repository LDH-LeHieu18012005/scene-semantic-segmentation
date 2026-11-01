$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'drawing_helpers.ps1')
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'kien_truc_dung_mau_paper/giai_thich_chi_tiet'
[IO.Directory]::CreateDirectory($out)|Out-Null
# Reuse only the drawing functions, without rerunning the architecture generator.
$tokens=$null;$errors=$null;$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'paper_format_figures.ps1'),[ref]$tokens,[ref]$errors)
$functions=$ast.FindAll({param($a) $a -is [Management.Automation.Language.FunctionDefinitionAst] -and $a.Name -in @('X','Seg','Arr','Label','OutlineBox','Tensor','CirclePlus')},$false)
foreach($f in $functions){Invoke-Expression $f.Extent.Text}
function StartFigure($w,$h){
 $script:bitmap=[Drawing.Bitmap]::new(($w*3),($h*3));$script:g=[Drawing.Graphics]::FromImage($bitmap);$g.Clear([Drawing.Color]::White);$g.ScaleTransform(3,3);$g.SmoothingMode='AntiAlias';$g.TextRenderingHint='AntiAliasGridFit'
 $script:svg=[Text.StringBuilder]::new();[void]$svg.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$($w*3)' height='$($h*3)' viewBox='0 0 $w $h'><rect width='$w' height='$h' fill='white'/>")
}
function FinishFigure($name){[void]$svg.AppendLine('</svg>');$bitmap.Save((Join-Path $out "$name.png"),[Drawing.Imaging.ImageFormat]::Png);[IO.File]::WriteAllText((Join-Path $out "$name.svg"),$svg.ToString(),[Text.UTF8Encoding]::new($false));$g.Dispose();$bitmap.Dispose()}
function Values($x,$y,$w,$h,$dx,$dy,$data,$n,$color){
 Tensor $x $y $w $h $dx $dy $color $n
 for($r=0;$r -lt $n;$r++){for($c=0;$c -lt $n;$c++){Label ([string]$data[$r*$n+$c]) ($x+$dx+($c+0.5)*$w/$n) ($y+$dy+($r+0.5)*$h/$n) 25 '#111111'}}
}
$red='#FF2638';$orange='#FF8518';$blue='#00A9E0';$purple='#A23479'

StartFigure 1500 630
Label 'Truyền đặc trưng từ nhánh 1×1 đến nhánh 2×2' 750 32 30
Label 'Ví dụ số trên một kênh trong 512 kênh; các giá trị mang tính minh họa.' 750 75 19
Values 65 145 75 75 17 12 @(4) 1 $red
Label 'z1: nguồn' 120 255 22;Label '1×1' 120 285 19
Arr 162 190 245 190;OutlineBox 'RESIZE' 250 174 100 32 21;Arr 350 190 420 190
Values 425 130 110 110 20 14 @(4,4,4,4) 2 $red
Label 'R(z1)' 500 285 22;Label '2×2' 500 313 19
Arr 558 190 620 190;OutlineBox '× tanh(w12)' 625 173 154 34 22;Arr 779 190 835 190
Label 'α12 = 0,25' 702 131 22 $purple
Values 840 130 110 110 20 14 @(1,1,1,1) 2 $red
Label 'Đóng góp có trọng số' 915 285 21
Arr 976 190 1080 190;CirclePlus 1090 190;Arr 1101 190 1180 190
Values 1185 130 110 110 20 14 @(2,3,4,5) 2 $orange
Label "z'2: đích sau cộng" 1260 285 22;Label 'Kích thước: 2×2' 1260 317 19
Values 875 385 110 110 20 14 @(1,2,3,4) 2 $orange
Label 'z2: đích ban đầu' 950 535 22
Seg 1006 451 1090 451;Arr 1090 451 1090 203
Label 'Cộng theo phần tử với đặc trưng đích' 1248 426 19
Label "z'2 = z2 + 0,25 × R(z1)" 750 575 29 $purple
Label 'Phép cộng được thực hiện theo vị trí và kênh. Ví dụ xét riêng cạnh 1→2.' 750 610 18
FinishFigure 'link_mot_canh_theo_mau_paper'

StartFigure 1500 680
Label 'Tổng hợp đặc trưng tại nhánh 2 từ hai nhánh lân cận' 750 32 29
Label 'Các tensor z biểu diễn đặc trưng ban đầu sau Pool → Conv 1×1 → BN → ReLU.' 750 76 19
Tensor 80 125 75 55 16 10 $red 1;Label 'z1: 1×1' 130 225 22
Arr 178 157 250 157;OutlineBox 'RESIZE → 2×2' 254 141 170 32 20;Arr 424 157 490 157
Tensor 495 115 90 90 17 12 $red 2;Arr 607 157 700 157;OutlineBox '× α12' 705 141 92 32 22
Seg 797 157 1030 157;Arr 1030 157 1030 314
Tensor 80 465 90 90 20 13 $blue 3;Label 'z3: 3×3' 140 604 22
Arr 195 516 250 516;OutlineBox 'POOL → 2×2' 254 500 170 32 20;Arr 424 516 490 516
Tensor 495 470 90 90 17 12 $blue 2;Arr 607 516 700 516;OutlineBox '× α32' 705 500 92 32 22
Seg 797 516 1030 516;Arr 1030 516 1030 340
Tensor 755 280 90 90 17 12 $orange 2;Label 'z2: đích ban đầu' 820 414 22
Arr 878 327 1018 327;CirclePlus 1030 327;Arr 1042 327 1150 327
Tensor 1155 280 90 90 17 12 $orange 2;Label "z'2: đích sau trộn" 1220 414 22
Label "z'2 = z2 + α12 R1→2(z1) + α32 R3→2(z3)" 750 640 28 $purple
Label 'Đặc trưng z′2 được nội suy về lưới backbone trước phép ghép kênh CONCAT.' 750 674 18
FinishFigure 'link_hai_nguon_vao_nhanh_2'
Write-Output "Created two PNG/SVG illustrations in $out"
