$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
. (Join-Path $PSScriptRoot 'drawing_helpers.ps1')
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$final=Join-Path $root 'slide_thuyet_trinh/THUYET_TRINH_LE_DUONG_HIEU_DUNG_MAU_PAPER.pptx'
$stage=Join-Path $root 'luu_tru/_bien_tap_slide'
if(Test-Path -LiteralPath $stage){throw 'Staging directory already exists'}
$stream=[IO.FileStream]::new($final,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite);$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Read);[IO.Compression.ZipFileExtensions]::ExtractToDirectory($zip,$stage);$zip.Dispose();$stream.Dispose()
$replacements=@{
 '02 / KIẾN TRÚC VÀ GIẢI THÍCH'='02 / KIẾN TRÚC VÀ CƠ CHẾ';
 'Hiểu link qua một cạnh: nhánh 1 → nhánh 2'='Cơ chế liên kết: truyền đặc trưng từ nhánh 1 đến nhánh 2';
 'Trong mô hình đầy đủ: nhánh 2 nhận hai nguồn'='Tổng hợp đặc trưng từ hai nhánh lân cận';
 'Vì sao chọn tanh và sáu cạnh kết nối?'='Cơ sở lựa chọn hàm tanh và cấu trúc sáu cạnh';
 'PPM 1236 → 1237: thay lưới pooling cuối'='Thay đổi lưới pooling: PPM 1236 và PPM 1237';
 '• Có dấu: cộng hoặc trừ đóng góp.'='• Hệ số có dấu điều chỉnh đóng góp đặc trưng.';
 '• w = 0 ⇒ α = 0: giữ PPM cơ sở'='• w = 0 ⇒ α = 0: bảo toàn PPM cơ sở';
 '  tại khởi tạo, rồi học mức chia sẻ.'='  tại khởi tạo; tối ưu hệ số khi huấn luyện.';
 '• Cấu trúc gọn: thêm 6 trọng số.'='• Chỉ bổ sung sáu trọng số vô hướng.';
 'Đây là lựa chọn thiết kế để khảo sát; chưa chứng minh tanh hoặc sáu cạnh là tối ưu.'='Chưa có thực nghiệm xác nhận tính tối ưu của hàm tanh hoặc cấu trúc sáu cạnh.';
 'Ảnh RGB → dự đoán nhãn cho từng pixel, kết hợp đặc trưng cục bộ và ngữ cảnh.'='Phân đoạn ngữ nghĩa: gán nhãn từng pixel dựa trên đặc trưng cục bộ và ngữ cảnh.';
 '• 20.210 ảnh train; 2.000 ảnh validation.'='• 20.210 ảnh huấn luyện; 2.000 ảnh kiểm định.';
 '• Giữ split chính thức, có 9 ảnh trùng RGB.'='• Giữ tập chia chính thức; ghi nhận 9 ảnh trùng RGB.';
 '• Chưa có kết quả test độc lập.'='• Chưa đánh giá trên tập kiểm tra độc lập.';
 '• Thay pooling, thêm link hoặc atrous.'='• Khảo sát pooling, liên kết nhánh và atrous.';
 'Ví dụ một kênh: nội suy 4 thành lưới 2×2, nhân α12=0,25 rồi cộng vào nhánh đích. Không nối thêm kênh.'='Ví dụ một kênh: nội suy về 2×2, nhân hệ số α12 = 0,25 và cộng theo phần tử; số kênh được bảo toàn.';
 'Nhánh 1 nội suy lên 2×2; nhánh 3 pooling xuống 2×2. Nhân hai hệ số riêng rồi cộng vào z2 ban đầu.'='Nhánh 1 được nội suy và nhánh 3 được pooling về 2×2; tổng hợp có trọng số với đặc trưng z2 ban đầu.';
 'Tham số train (M)'='Tham số (triệu)';
 'Median (ms)'='Trung vị (ms)';
 'Median T4 (ms)'='Trung vị T4 (ms)';
 'Pixel acc. (%)'='Độ chính xác pixel (%)';
 'Thảo luận và giải đáp'='Trao đổi học thuật'
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $stage 'ppt/slides') -Filter 'slide*.xml'){
 $x=ReadXml $file.FullName;$n=NS $x
 foreach($t in $x.SelectNodes('//a:t',$n)){if($replacements.ContainsKey($t.InnerText)){$t.InnerText=$replacements[$t.InnerText]}}
 # Establish consistent typography while respecting the existing box geometry.
 foreach($shape in $x.SelectNodes('//p:sp[p:txBody]',$n)){
  $text=($shape.SelectNodes('.//a:t',$n)|ForEach-Object{$_.InnerText}) -join ' '
  $off=$shape.SelectSingleNode('p:spPr/a:xfrm/a:off',$n);$ext=$shape.SelectSingleNode('p:spPr/a:xfrm/a:ext',$n)
  $y=if($off){[long]$off.y}else{0};$h=if($ext){[long]$ext.cy}else{0};$w=if($ext){[long]$ext.cx}else{0}
  $isTitle=($y -ge 500000 -and $y -le 1150000 -and $w -gt 11000000)
  $isFooter=($y -gt 9600000);$isSection=($y -lt 500000 -and $w -gt 8000000)
  foreach($rpr in $shape.SelectNodes('.//a:rPr | .//a:defRPr | .//a:endParaRPr',$n)){
   foreach($tag in @('latin','ea','cs')){$font=$rpr.SelectSingleNode("a:$tag",$n);if(!$font){$font=$x.CreateElement('a',$tag,$n.LookupNamespace('a'));$rpr.AppendChild($font)|Out-Null};$font.SetAttribute('typeface','Arial')}
   if($file.Name -ne 'slide1.xml' -and $file.Name -ne 'slide22.xml'){
    $old=if($rpr.HasAttribute('sz')){[int]$rpr.sz}else{2200}
    if($isTitle){$size=if($text.Length -gt 65){3000}else{3400};$rpr.SetAttribute('b','1')}
    elseif($isFooter){$size=1400}
    elseif($isSection){$size=1800}
    elseif($h -lt 420000){$size=[Math]::Min([Math]::Max($old,1800),2200)}
    else{$size=[Math]::Min([Math]::Max($old,2200),3000)}
    $rpr.SetAttribute('sz',"$size")
   }
  }
 }
 $x.Save($file.FullName)
}
# Academic wording inside the two explanatory illustrations.
$generator=Join-Path $PSScriptRoot 'minh_hoa_link.ps1';$code=[IO.File]::ReadAllText($generator,[Text.Encoding]::UTF8)
$figureText=@{
 'Một link: nhánh 1×1 gửi thông tin sang nhánh 2×2'='Truyền đặc trưng từ nhánh 1×1 đến nhánh 2×2';
 'Minh họa một kênh trong 512 kênh; các số chỉ dùng giải thích phép tính.'='Ví dụ số trên một kênh trong 512 kênh; các giá trị mang tính minh họa.';
 'Thông tin gửi sang'='Đóng góp có trọng số';
 'Vẫn là 2×2'='Kích thước: 2×2';
 'Giữ đặc trưng đích rồi cộng thêm'='Cộng theo phần tử với đặc trưng đích';
 'Đây là phép cộng theo vị trí và kênh; không nối thêm kênh. Tách riêng cạnh 1→2 để dễ nhìn.'='Phép cộng được thực hiện theo vị trí và kênh. Ví dụ xét riêng cạnh 1→2.';
 'Trong mô hình đầy đủ, nhánh 2 nhận từ cả nhánh 1 và nhánh 3'='Tổng hợp đặc trưng tại nhánh 2 từ hai nhánh lân cận';
 'Các z đều là đặc trưng ban đầu sau Pool → Conv 1×1 → BN → ReLU.'='Các tensor z biểu diễn đặc trưng ban đầu sau Pool → Conv 1×1 → BN → ReLU.';
 'Sau đó mới nội suy z′2 về lưới backbone để ghép CONCAT với các nhánh khác.'='Đặc trưng z′2 được nội suy về lưới backbone trước phép ghép kênh CONCAT.'
}
foreach($key in $figureText.Keys){$code=$code.Replace($key,$figureText[$key])};[IO.File]::WriteAllText($generator,$code,[Text.UTF8Encoding]::new($true));& $generator
foreach($v in @(@(30,'link_mot_canh_theo_mau_paper.png'),@(31,'link_hai_nguon_vao_nhanh_2.png'))){$rp=Join-Path $stage "ppt/slides/_rels/slide$($v[0]).xml.rels";$r=ReadXml $rp;$rel=$r.DocumentElement.ChildNodes|Where-Object{$_.Type.EndsWith('/image')};$dest=[IO.Path]::GetFullPath((Join-Path (Join-Path $stage 'ppt/slides') $rel.Target));Copy-Item -LiteralPath (Join-Path $root "kien_truc_dung_mau_paper/giai_thich_chi_tiet/$($v[1])") -Destination $dest -Force}
function Esc($s){[Security.SecurityElement]::Escape($s)}
function TextBox($id,$text,$x,$y,$w,$h,$sz,$bold=$false){$b=if($bold){' b="1"'}else{''};$t=Esc $text;return "<p:sp><p:nvSpPr><p:cNvPr id='$id' name='Text $id'/><p:cNvSpPr txBox='1'/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x='$x' y='$y'/><a:ext cx='$w' cy='$h'/></a:xfrm><a:prstGeom prst='rect'><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap='square' lIns='0' rIns='0' tIns='0' bIns='0'/><a:lstStyle/><a:p><a:r><a:rPr lang='vi-VN' sz='$sz'$b><a:solidFill><a:srgbClr val='172B4D'/></a:solidFill><a:latin typeface='Arial'/></a:rPr><a:t>$t</a:t></a:r></a:p></p:txBody></p:sp>"}
$body=(TextBox 2 '03 / ỨNG DỤNG MINH HỌA' 650000 200000 16000000 400000 1800)+(TextBox 3 'PSP Studio: giao diện và kết quả phân đoạn' 650000 720000 17000000 650000 3400 $true)
$body+="<p:pic><p:nvPicPr><p:cNvPr id='4' name='Giao diện PSP Studio'/><p:cNvPicPr><a:picLocks noChangeAspect='1'/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed='rIdDemo'/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x='700000' y='1700000'/><a:ext cx='9800000' cy='7486111'/></a:xfrm><a:prstGeom prst='rect'><a:avLst/></a:prstGeom></p:spPr></p:pic>"
$body+=(TextBox 5 'Cấu hình suy luận' 11100000 1900000 6300000 600000 2800 $true)
$body+=(TextBox 6 'PSPNet–ResNet-101, PPM 1237; 150 lớp ADE20K. Suy luận trên CPU.' 11100000 2800000 6100000 1400000 2600)
$body+=(TextBox 7 'Kết quả hiển thị' 11100000 4500000 6300000 600000 2800 $true)
$body+=(TextBox 8 'Ảnh đầu vào, lớp phủ phân đoạn và kích hoạt đặc trưng trung gian.' 11100000 5400000 6100000 1400000 2600)
$body+=(TextBox 9 'Phạm vi minh họa' 11100000 7100000 6300000 600000 2800 $true)
$body+=(TextBox 10 'Ảnh chụp từ phiên kiểm tra đã lưu. Bản đồ kích hoạt không phải attention hoặc nhãn phân đoạn.' 11100000 7950000 6100000 1500000 2300)
$body+=(TextBox 11 'Lê Dương Hiếu · 143011 | Thị giác máy tính và ứng dụng' 650000 9850000 15000000 300000 1400)+(TextBox 12 '14' 17000000 9850000 500000 300000 1400)
$xml="<?xml version='1.0' encoding='utf-8'?><p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val='FFFFFF'/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id='1' name=''/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>$body</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
[IO.File]::WriteAllText((Join-Path $stage 'ppt/slides/slide38.xml'),$xml,[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $root 'demo/validation/desktop-result.png') -Destination (Join-Path $stage 'ppt/media/pspstudio_demo.png')
$rxml="<?xml version='1.0' encoding='utf-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rIdDemo' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/image' Target='../media/pspstudio_demo.png'/><Relationship Id='rIdLayout' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout' Target='../slideLayouts/slideLayout1.xml'/></Relationships>"
[IO.File]::WriteAllText((Join-Path $stage 'ppt/slides/_rels/slide38.xml.rels'),$rxml,[Text.UTF8Encoding]::new($false))
$pp=Join-Path $stage 'ppt/presentation.xml';$pr=ReadXml $pp;$n=NS $pr;$list=$pr.SelectSingleNode('//p:sldIdLst',$n);$anchor=$list.ChildNodes[13];$node=$pr.CreateElement('p','sldId',$n.LookupNamespace('p'));$node.SetAttribute('id','1638');$node.SetAttribute('id',$n.LookupNamespace('r'),'rIdDemo38')|Out-Null;$list.InsertBefore($node,$anchor)|Out-Null
$rp=Join-Path $stage 'ppt/_rels/presentation.xml.rels';$rs=ReadXml $rp;$r=$rs.CreateElement('Relationship',$rs.DocumentElement.NamespaceURI);$r.SetAttribute('Id','rIdDemo38');$r.SetAttribute('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide');$r.SetAttribute('Target','slides/slide38.xml');$rs.DocumentElement.AppendChild($r)|Out-Null
$ctp=Join-Path $stage '[Content_Types].xml';$ct=ReadXml $ctp;$o=$ct.CreateElement('Override',$ct.DocumentElement.NamespaceURI);$o.SetAttribute('PartName','/ppt/slides/slide38.xml');$o.SetAttribute('ContentType','application/vnd.openxmlformats-officedocument.presentationml.slide+xml');$ct.DocumentElement.AppendChild($o)|Out-Null
$page=0;foreach($id in $list.ChildNodes){$page++;$rid=$id.GetAttribute('id',$n.LookupNamespace('r'));$r=$rs.DocumentElement.ChildNodes|Where-Object{$_.Id -eq $rid};$path=Join-Path (Join-Path $stage 'ppt') $r.Target;$s=ReadXml $path;$sn=NS $s;$ts=$s.SelectNodes('//a:t',$sn);if($ts[$ts.Count-1].InnerText -match '^\d+$'){$ts[$ts.Count-1].InnerText="$page"};$s.Save($path)}
$pr.Save($pp);$rs.Save($rp);$ct.Save($ctp);$app=ReadXml (Join-Path $stage 'docProps/app.xml');$app.Properties.Slides='18';$app.Save((Join-Path $stage 'docProps/app.xml'))
$xmlCount=0;$relCount=0
foreach($file in Get-ChildItem -LiteralPath $stage -Recurse -File){if($file.Extension -in @('.xml','.rels')){$x=[xml]::new();$x.Load($file.FullName);$xmlCount++;if($file.Extension -eq '.rels'){$dir=Split-Path (Split-Path $file.FullName -Parent) -Parent;foreach($r in $x.DocumentElement.ChildNodes){if($r.LocalName -ne 'Relationship' -or $r.TargetMode -eq 'External'){continue};$relCount++;$target=if($r.Target.StartsWith('/')){Join-Path $stage $r.Target.TrimStart('/')}else{[IO.Path]::GetFullPath((Join-Path $dir $r.Target))};if(!(Test-Path -LiteralPath $target)){throw "Unresolved relationship: $target"}}}}}
$new=Join-Path $root 'slide_thuyet_trinh/.bien_tap.pptx';$stream=[IO.FileStream]::new($new,[IO.FileMode]::CreateNew);$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create);foreach($file in Get-ChildItem -LiteralPath $stage -Recurse -File){$name=$file.FullName.Substring($stage.Length+1).Replace('\','/');[IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$file.FullName,$name)|Out-Null};$zip.Dispose();$stream.Dispose()
[IO.File]::Copy($new,$final,$true);Remove-Item -LiteralPath $new
Write-Output "Updated current presentation: 18 slides; $xmlCount XML parts; $relCount local relationships verified."
