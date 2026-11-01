$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$file = Join-Path $root 'slide_thuyet_trinh/THUYET_TRINH_LE_DUONG_HIEU_DUNG_MAU_PAPER.pptx'
$backup = Join-Path $root 'luu_tru/THUYET_TRINH_TRUOC_CAP_NHAT_KHAI_NIEM_29_10_2026.pptx'
$stage = Join-Path $root ('luu_tru/ppt_update_' + [guid]::NewGuid().ToString('N'))
if (!(Test-Path -LiteralPath $backup)) { Copy-Item -LiteralPath $file -Destination $backup }
[IO.Compression.ZipFile]::ExtractToDirectory($file, $stage)
function LoadXml($path) { $x = [xml]::new(); $x.PreserveWhitespace=$true; $x.Load($path); return ,$x }
function Names($x) {
 $n=[Xml.XmlNamespaceManager]::new($x.NameTable)
 $n.AddNamespace('p','http://schemas.openxmlformats.org/presentationml/2006/main')
 $n.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main')
 $n.AddNamespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
 return ,$n
}
function SetText($x,$id,$lines,$size=2600) {
 $n=Names $x
 $shape=$x.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@id='$id']]",$n)
 $body=$shape.SelectSingleNode('p:txBody',$n)
 $template=$body.SelectSingleNode('a:p',$n).CloneNode($true)
 foreach($p in @($body.SelectNodes('a:p',$n))) { [void]$body.RemoveChild($p) }
 foreach($line in $lines) {
  $p=$template.CloneNode($true)
  $p.SelectSingleNode('.//a:t',$n).InnerText=$line
  $p.SelectSingleNode('.//a:rPr',$n).SetAttribute('sz',"$size")
  $p.SelectSingleNode('a:pPr/a:spcAft/a:spcPts',$n).SetAttribute('val','900')
  [void]$body.AppendChild($p)
 }
}
$templatePath=Join-Path $stage 'ppt/slides/slide9.xml'
$specs=@(
 @{after=3; title='Backbone, baseline và vai trò của PPM'; left='ResNet-101 · bộ trích xuất đặc trưng'; right='PPM · bổ sung ngữ cảnh đa tỉ lệ';
 l=@('• Deep stem: ba convolution 3×3.','• Residual: y = F(x) + x.','• Dilation tầng cuối: 2 và 4; OS8.','• 473×473 → 2048×60×60.');
 r=@('• Pooling ra lưới 1, 2, 3, 6 (hoặc 7).','• Mỗi nhánh: Conv 1×1 → 512 kênh.','• Nội suy về 60×60 rồi nối với F.','• 2048 + 4×512 = 4096 kênh.');
 bottom=@('Baseline b6 = ResNet-101 + PPM 1236 + head phân đoạn 150 lớp.','PPM kết hợp ngữ cảnh toàn ảnh và từng vùng để hỗ trợ nhận diện pixel.');
 note='Bins là kích thước đầu ra pooling; b7 chỉ đổi lưới cuối 6×6 thành 7×7.'},
 @{after=8; title='Atrous convolution: mở rộng vùng quan sát'; left='Cơ chế tích chập giãn'; right='Ba nhánh bổ sung trong đồ án';
 l=@('• Dilation r tăng khoảng cách lấy mẫu.','• Phạm vi: k_eff = k + (k − 1)(r − 1).','• Kernel 3×3, r=2: phủ vùng 5×5.','• Vẫn có 9 trọng số / cặp kênh.');
 r=@('• Cùng nhận F: 2048×60×60.','• Conv 3×3; r = 12, 24, 36.','• Mỗi nhánh: 256 kênh + BN + ReLU.','• Stride 1; padding r: giữ kích thước.');
 bottom=@('Nối PPM (4096) và ba nhánh (3×256) → 4864 kênh → head phân đoạn.','Dilation mở rộng vùng lấy mẫu; thêm nhánh làm tăng tham số và MACs.');
 note='Biến thể _aspp giữ nguyên PPM và bổ sung atrous theo ý tưởng ASPP của DeepLabV3.'},
 @{after=9; title='Phương pháp huấn luyện: fine-tune toàn mạng'; left='Khởi tạo từ mô hình pretrained'; right='Học có giám sát trên ADE20K';
 l=@('• Backbone đã pretrain trên ImageNet.','• Dùng trọng số resnet101_v2.pth.','• PPM và head được khởi tạo mới.','• Bỏ head phân loại ImageNet.');
 r=@('• Đầu vào: ảnh và mask nhãn pixel.','• Backbone và head đều nhận gradient.','• SGD cập nhật toàn bộ tham số.','• Không chỉ huấn luyện riêng head.');
 bottom=@('Pretraining: giai đoạn học trước; đồ án sử dụng trọng số có sẵn.','Supervised fine-tuning: học từ nhãn đích. Full fine-tuning: cập nhật toàn mạng.');
 note='Cách gọi phù hợp: transfer learning và supervised full fine-tuning cho phân đoạn ngữ nghĩa.'},
 @{after=9; title='Batch size, epoch và optimizer step'; left='Ba đơn vị cần phân biệt'; right='Áp dụng cấu hình Modal';
 l=@('• Batch size: số ảnh trong một batch.','• Epoch: một lượt qua tập train.','• Step: một lần optimizer cập nhật.','• Không tích lũy: 1 batch = 1 step.');
 r=@('• N = 20.210; batch thực B = 16.','• Bỏ batch cuối thiếu ảnh.','• floor(N/B) = 1.263 step/epoch.','• 10.000 step ≈ 7,92 epoch.');
 bottom=@('Kaggle: micro-batch 2 ảnh × tích lũy gradient 8 lần = batch hiệu dụng 16.','Một optimizer step gồm 8 micro-batch; 10.000 step xử lý 160.000 lượt ảnh.');
 note='Giữ số epoch: batch lớn → ít step. Giữ số step: batch lớn → nhiều lượt ảnh hơn.'},
 @{after=9; title='Cross entropy và giám sát ở nhánh phụ'; left='Loss cho từng pixel'; right='Loss tổng khi huấn luyện';
 l=@('• Pixel có lớp đúng y: L = −ln(p_y).','• p_y=0,9 → L≈0,105.','• p_y=0,1 → L≈2,303.','• Trung bình trên các pixel hợp lệ.');
 r=@('• L_total = L_main + 0,4 × L_aux.','• Nhánh phụ lấy đầu ra layer3.','• Bổ sung gradient cho tầng giữa.','• Suy luận chỉ chạy nhánh chính.');
 bottom=@('CrossEntropyLoss nhận logits trực tiếp; không softmax trước khi tính loss.','ADE20K: nhãn 1–150 → 0–149; nhãn gốc 0 và padding → 255 (bỏ qua).');
 note='Loss dùng để tối ưu trọng số; mIoU dùng để đánh giá chất lượng phân đoạn.'},
 @{after=9; title='Siêu tham số điều khiển quá trình học'; left='Tối ưu trọng số · Modal'; right='Dữ liệu và điều chuẩn · Modal';
 l=@('• SGD; backbone LR ban đầu 0,01.','• Head LR ×10: ban đầu 0,1.','• Momentum 0,9: tích lũy hướng học.','• Weight decay 10⁻⁴: phạt trọng số.');
 r=@('• Crop 473; batch 16; 10.000 step.','• Scale 0,5–2; xoay ±10°; lật ảnh.','• Dropout2d 0,1; aux weight 0,4.','• Seed 17; train AMP BF16.');
 bottom=@('Lịch poly: lr(t) = lr₀ × (1 − t/T)^0,9; giảm bước cập nhật về cuối.','Kaggle: backbone LR 0,001; head ×10; warmup 500 step; clip norm 1,0.');
 note='Siêu tham số do người thiết kế đặt; trọng số convolution là tham số được học từ dữ liệu.'},
 @{after=9; title='Metrics và chi phí: đọc đúng các chỉ số'; left='Chất lượng phân đoạn'; right='Chi phí tính toán và thời gian';
 l=@('• IoU = TP / (TP + FP + FN).','• mIoU: trung bình IoU các lớp.','• Pixel accuracy: tỉ lệ pixel đúng.','• Bỏ pixel 255; cộng confusion matrix.');
 r=@('• MAC: một phép nhân–cộng dồn.','• 1 GMAC = 10⁹ MAC; FLOPs ≈ 2 MACs.','• Params: số trọng số của mạng.','• Latency / VRAM: thời gian / bộ nhớ.');
 bottom=@('MACs Conv = B×H_out×W_out×C_out×(C_in/groups)×K_h×K_w.','Benchmark: chỉ Conv/Linear; bỏ pooling, BN, ReLU, nội suy; không chạy aux.');
 note='mIoU tính từ tổng confusion matrix; so MACs phải cùng đầu vào. Ít MACs chưa chắc nhanh hơn.'},
 @{after=9; title='Pipeline xử lý ảnh: train và suy luận'; left='Huấn luyện: ảnh đi cùng mask'; right='Suy luận và đánh giá';
 l=@('1. Đọc RGB; đổi mã nhãn ADE20K.','2. Scale/xoay/lật đồng bộ ảnh–mask.','3. Crop 473; chuẩn hóa ImageNet.','4. Backbone → PPM → head(s).');
 r=@('1. Cạnh dài 512; giữ tỉ lệ ảnh.','2. Chuẩn hóa; cửa sổ 473 chồng lấn.','3. Forward FP32 → softmax.','4. Trung bình xác suất vùng chồng.');
 bottom=@('Train: loss chính + 0,4 loss phụ → backward → optimizer step.','Suy luận: bỏ padding → resize xác suất về ảnh gốc → argmax → mask màu.');
 note='Ảnh nội suy bilinear; mask nhãn dùng nearest. Đánh giá cuối: 2.000 ảnh, một tỉ lệ, không flip.'}
)
$pres=LoadXml (Join-Path $stage 'ppt/presentation.xml'); $pn=Names $pres
$rels=LoadXml (Join-Path $stage 'ppt/_rels/presentation.xml.rels')
$types=LoadXml (Join-Path $stage '[Content_Types].xml')
$original=@($pres.SelectNodes('//p:sldIdLst/p:sldId',$pn))
$list=$pres.SelectSingleNode('//p:sldIdLst',$pn)
$k=0
foreach($s in $specs) {
 $k++; $num=15+$k; $x=LoadXml $templatePath
 SetText $x 2 @('02 / CONCEPTS ET PROTOCOLE'.Replace('CONCEPTS ET PROTOCOLE','CƠ CHẾ VÀ HUẤN LUYỆN')) 1800
 SetText $x 3 @($s.title) 3400
 SetText $x 6 @($s.left) 2800
 SetText $x 8 @($s.right) 2800
 SetText $x 7 $s.l 2600; SetText $x 9 $s.r 2600
 SetText $x 10 $s.bottom 2700; SetText $x 11 @($s.note) 2500
 # Taller panels accommodate four short points with consistent body typography.
 $n=Names $x
 foreach($id in @(4,5)) { $x.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@id='$id']]/p:spPr/a:xfrm/a:ext",$n).SetAttribute('cy','3900000') }
 foreach($id in @(7,9)) { $x.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@id='$id']]/p:spPr/a:xfrm/a:ext",$n).SetAttribute('cy','2400000') }
 $x.Save((Join-Path $stage "ppt/slides/slide$num.xml"))
 Copy-Item -LiteralPath (Join-Path $stage 'ppt/slides/_rels/slide9.xml.rels') -Destination (Join-Path $stage "ppt/slides/_rels/slide$num.xml.rels")
 $rid="rIdConcept$k"
 $rel=$rels.CreateElement('Relationship',$rels.DocumentElement.NamespaceURI)
 $rel.SetAttribute('Id',$rid);$rel.SetAttribute('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide');$rel.SetAttribute('Target',"slides/slide$num.xml");[void]$rels.DocumentElement.AppendChild($rel)
 $ct=$types.CreateElement('Override',$types.DocumentElement.NamespaceURI);$ct.SetAttribute('PartName',"/ppt/slides/slide$num.xml");$ct.SetAttribute('ContentType','application/vnd.openxmlformats-officedocument.presentationml.slide+xml');[void]$types.DocumentElement.AppendChild($ct)
 $node=$pres.CreateElement('p','sldId',$pn.LookupNamespace('p'));$node.SetAttribute('id',"$(2000+$k)");[void]$node.SetAttribute('id',$pn.LookupNamespace('r'),$rid)
 [void]$list.InsertBefore($node,$original[$s.after])
}
$pres.Save((Join-Path $stage 'ppt/presentation.xml'));$rels.Save((Join-Path $stage 'ppt/_rels/presentation.xml.rels'));$types.Save((Join-Path $stage '[Content_Types].xml'))
# Clarify the existing protocol slide without changing its experiment results.
$x=LoadXml $templatePath
SetText $x 9 @('• ResNet-50; crop 384.','• Batch hiệu dụng 16 = 2 ảnh × 8 lần.','• Cố định thống kê BN; train FP16.') 2700
SetText $x 10 @('Checkpoint sau 10.000 optimizer updates; Modal ≈ 7,92 epoch; đánh giá FP32.','Resize cạnh dài 512 + cửa sổ trượt; 2.000 ảnh validation; không flip.') 2700
$x.Save($templatePath)
# Renumber every existing footer according to the final presentation order.
$index=0
foreach($node in $pres.SelectNodes('//p:sldIdLst/p:sldId',$pn)) {
 $index++;$rid=$node.GetAttribute('id',$pn.LookupNamespace('r'))
 $rel=@($rels.DocumentElement.ChildNodes | Where-Object { $_.GetAttribute('Id') -eq $rid })[0]
 $path=Join-Path $stage ('ppt/'+$rel.GetAttribute('Target'))
 $x=LoadXml $path;$n=Names $x
 foreach($shape in $x.SelectNodes('//p:sp[p:txBody]',$n)) {
  $off=$shape.SelectSingleNode('p:spPr/a:xfrm/a:off',$n)
  $texts=$shape.SelectNodes('.//a:t',$n)
  if($off -and [long]$off.y -gt 9600000 -and [long]$off.x -ge 16500000 -and $texts.Count -eq 1 -and $texts[0].InnerText -match '^\d+$') { $texts[0].InnerText="$index" }
 }
 $x.Save($path)
}
$appPath=Join-Path $stage 'docProps/app.xml';$app=LoadXml $appPath
$sn=$app.SelectSingleNode('//*[local-name()="Slides"]');if($sn){$sn.InnerText="$index"};$app.Save($appPath)
$out=Join-Path $root ('slide_thuyet_trinh/THUYET_TRINH_CAP_NHAT_'+[guid]::NewGuid().ToString('N')+'.tmp.pptx')
if(Test-Path -LiteralPath $out){throw 'Temporary output already exists'}
$archive=[IO.Compression.ZipFile]::Open($out,[IO.Compression.ZipArchiveMode]::Create)
foreach($part in Get-ChildItem -LiteralPath $stage -Recurse -File) {
 $entryName=$part.FullName.Substring($stage.Length+1).Replace('\','/')
 [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$part.FullName,$entryName)
}
$archive.Dispose()
# Validate XML, internal relationships, slide count and content before replacing.
$z=[IO.Compression.ZipFile]::OpenRead($out)
$xmlCount=0
foreach($entry in $z.Entries) {
 if($entry.FullName -match '\.(xml|rels)$') {
  $reader=[IO.StreamReader]::new($entry.Open());$doc=[xml]$reader.ReadToEnd();$reader.Dispose();$xmlCount++
  if($entry.FullName -match '\.rels$') {
   foreach($rel in $doc.DocumentElement.ChildNodes) {
    if($rel.NodeType -ne 'Element' -or $rel.GetAttribute('TargetMode') -eq 'External'){continue}
    $base=if($entry.FullName -eq '_rels/.rels'){''}else{($entry.FullName -replace '_rels/[^/]+\.rels$','')}
    $uri=[Uri]::new([Uri]('http://package/'+$base),$rel.GetAttribute('Target'))
    $target=[Uri]::UnescapeDataString($uri.AbsolutePath.TrimStart('/'))
    if(!$z.GetEntry($target)){throw "Missing relationship target: $target"}
   }
  }
 }
}
$slideCount=@($z.Entries | Where-Object FullName -Match '^ppt/slides/slide\d+\.xml$').Count
if($slideCount -ne 23){throw "Unexpected slide count $slideCount"}
$z.Dispose()
Copy-Item -LiteralPath $out -Destination $file -Force
Write-Output "Updated: $file"
Write-Output "Backup: $backup"
Write-Output "Validated $slideCount slides, $xmlCount XML parts and all internal relationship targets."
