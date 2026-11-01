$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$file=Join-Path $root 'slide_thuyet_trinh/THUYET_TRINH_LE_DUONG_HIEU_DUNG_MAU_PAPER.pptx'
$tmp=Join-Path ([IO.Path]::GetTempPath()) ('ppt_simplify_'+[guid]::NewGuid().ToString('N')+'.pptx')
Copy-Item -LiteralPath $file -Destination $tmp
$zip=[IO.Compression.ZipFile]::Open($tmp,[IO.Compression.ZipArchiveMode]::Update)
function ReadXml($name){$reader=[IO.StreamReader]::new($zip.GetEntry($name).Open());$x=[xml]::new();$x.PreserveWhitespace=$true;$x.LoadXml($reader.ReadToEnd());$reader.Dispose();return ,$x}
function NS($x){$n=[Xml.XmlNamespaceManager]::new($x.NameTable);$n.AddNamespace('p','http://schemas.openxmlformats.org/presentationml/2006/main');$n.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main');return ,$n}
function Text($x,$id,$lines){
 $n=NS $x;$shape=$x.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@id='$id']]",$n)
 $body=$shape.SelectSingleNode('p:txBody',$n);$template=$body.SelectSingleNode('a:p',$n).CloneNode($true)
 foreach($p in @($body.SelectNodes('a:p',$n))){[void]$body.RemoveChild($p)}
 foreach($line in $lines){$p=$template.CloneNode($true);$runs=@($p.SelectNodes('a:r',$n));foreach($r in $runs | Select-Object -Skip 1){[void]$p.RemoveChild($r)};$p.SelectSingleNode('.//a:t',$n).InnerText=$line;[void]$body.AppendChild($p)}
}
function WriteXml($name,$x){$zip.GetEntry($name).Delete();$e=$zip.CreateEntry($name);$stream=$e.Open();$writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false));$writer.Write($x.OuterXml);$writer.Dispose()}
foreach($i in 1..22){
 $name="ppt/slides/slide$i.xml";$x=ReadXml $name;$changed=$false
 if($i -eq 13){
  Text $x 3 @('Ngân sách huấn luyện: các ý chính')
  Text $x 6 @('Hiểu ngắn gọn')
  Text $x 7 @('• Batch size: số ảnh xử lý mỗi batch.','• Step: một lần cập nhật trọng số.','• Epoch: một lượt qua tập train.')
  Text $x 8 @('Cấu hình sử dụng')
  Text $x 9 @('• Modal: batch thực 16 ảnh.','• Kaggle: 2 ảnh × tích lũy 8 lần.','• Mỗi mô hình: 10.000 lần cập nhật.')
  Text $x 10 @('Hai nhóm cùng batch hiệu dụng 16 ảnh và ngân sách 10.000 lần cập nhật.')
  Text $x 11 @('Batch hiệu dụng giống nhau; cách xử lý BatchNorm của hai nhóm khác nhau.')
  $changed=$true
 }
 if($i -eq 14){
  Text $x 3 @('Cross entropy: đo sai số để mô hình học')
  Text $x 6 @('Hàm mất mát (loss)')
  Text $x 7 @('• So dự đoán với nhãn đúng từng pixel.','• Dự đoán đúng, tự tin → loss nhỏ.','• Dự đoán sai, tự tin → loss lớn.')
  Text $x 8 @('Nhánh chính và nhánh phụ')
  Text $x 9 @('• Nhánh chính tạo kết quả phân đoạn.','• Nhánh phụ hỗ trợ khi huấn luyện.','• Suy luận chỉ chạy nhánh chính.')
  Text $x 10 @('Mô hình cập nhật trọng số để giảm loss; bỏ qua các pixel không có nhãn hợp lệ.')
  Text $x 11 @('Loss hướng dẫn quá trình học; mIoU đánh giá chất lượng phân đoạn.')
  $changed=$true
 }
 if($i -eq 15){
  Text $x 3 @('Đánh giá chất lượng và chi phí mô hình')
  Text $x 6 @('Chất lượng phân đoạn')
  Text $x 7 @('• IoU: mức trùng khớp với nhãn thật.','• mIoU: trung bình IoU của các lớp.','• Pixel accuracy: tỉ lệ pixel đúng.','• Chỉ số chất lượng cao hơn là tốt hơn.')
  Text $x 8 @('Chi phí và tốc độ')
  Text $x 9 @('• MACs: số phép nhân–cộng.','• 1 GMAC = 1 tỷ phép nhân–cộng.','• Tham số: số giá trị mô hình học.','• Latency: thời gian chạy thực tế.')
  Text $x 10 @('Cùng kích thước đầu vào: MACs cao hơn nghĩa là cần nhiều phép tính hơn.')
  Text $x 11 @('MACs đo lượng tính toán; latency đo thời gian. Ít MACs chưa chắc chạy nhanh hơn.')
  $changed=$true
 }
 $n=NS $x
 foreach($s in $x.SelectNodes('//p:sp[p:txBody]',$n)){
  $off=$s.SelectSingleNode('p:spPr/a:xfrm/a:off',$n);$ts=$s.SelectNodes('.//a:t',$n)
  if($off -and [long]$off.y -gt 9600000 -and [long]$off.x -ge 16500000 -and $ts.Count -eq 1 -and $ts[0].InnerText -match '^\d+$' -and $ts[0].InnerText -ne "$i"){$ts[0].InnerText="$i";$changed=$true}
 }
 if($changed){WriteXml $name $x}
}
$zip.Dispose()
$check=[IO.Compression.ZipFile]::OpenRead($tmp);$parts=0
foreach($e in $check.Entries){if($e.FullName -match '\.(xml|rels)$'){$reader=[IO.StreamReader]::new($e.Open());$parsed=[xml]$reader.ReadToEnd();$reader.Dispose();$parts++}}
$slides=@($check.Entries | Where-Object FullName -Match '^ppt/slides/slide\d+\.xml$').Count
if($slides -ne 22){throw 'Unexpected slide count'}
$check.Dispose()
Copy-Item -LiteralPath $tmp -Destination $file -Force
Remove-Item -LiteralPath $tmp
Write-Output "Updated slides 13, 14, 15; corrected footer numbering. Validated $slides slides and $parts XML parts."
