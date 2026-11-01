# Rà soát thực nghiệm các notebook ADE20K

Ngày rà soát: 24/10/2026.

**Phạm vi đã thống nhất sau rà soát:** dừng chạy thực nghiệm mới do giới hạn tài nguyên. Tập trung phân tích kết quả đã lưu. Các gợi ý train bổ sung ở mục 3 chỉ là hướng phát triển tương lai, không phải điều kiện bắt buộc để hoàn thành báo cáo hiện tại.

Phạm vi: cả 6 notebook trong thư mục, mã runner nhúng, output văn bản đã lưu, README và mã Python liên quan. Đây là rà soát tĩnh; không chạy lại huấn luyện hoặc xác thực checkpoint trên GPU. Thư mục hiện không chứa các checkpoint, CSV/NPZ kết quả gốc hay dataset để tính lại chỉ số. Các số dưới đây được chép và tính chênh lệch từ output notebook, chưa phải kết quả tái lập độc lập.

## 1. Đề tài và kết quả đang có

Đề tài hiện xoay quanh phân đoạn ngữ nghĩa ADE20K 150 lớp bằng PSPNet: thay PPM {1,2,3,6} bằng {1,2,3,7}, thêm liên kết học được giữa các nhánh PPM, hoặc thêm ba nhánh atrous convolution. FCN và DeepLabV3 là các đối chứng trên Kaggle.

Nhóm Modal dùng ResNet-101 deep stem, crop 473, batch thực 16, BF16 khi train, BN học running statistics, seed 17, 10.000 optimizer updates. Đánh giá FP32, resize cạnh dài 512 rồi sliding window, không flip, đủ 2.000 ảnh validation.

| Biến thể Modal | mIoU (%) | Chênh lệch so với cùng bins, không mở rộng (điểm %) |
|---|---:|---:|
| b6: PPM 1236 | 34,4546 | — |
| b7: PPM 1237 | 35,7903 | — |
| b6_link | 35,0606 | +0,6060 |
| b7_link | 35,1730 | -0,6173 |
| b6_aspp | 36,2776 | +1,8230 |
| b7_aspp | 36,2284 | +0,4381 |

Các chênh lệch mở rộng so với bản gốc trong bảng là phép trừ số đã lưu; chưa có CI trực tiếp cho các cặp đó trong output được rà soát. Với ASPP, cần kiểm tra đồng nhất phần trọng số chung giữa thí nghiệm v4 và v5 trước khi coi đó là so sánh ghép cặp init.

Các bootstrap đã thực hiện:

| Cặp so sánh | Δ mIoU (điểm %) | CI 95% trên ảnh validation |
|---|---:|---:|
| b7 − b6 | +1,3357 | [+0,7747; +1,8616] |
| b7_link − b6_link | +0,1124 | [-0,5170; +0,5748] |
| b7_aspp − b6_aspp | -0,0492 | [-0,4204; +0,4196] |

Diễn giải: lợi thế 1237 quan sát được ở PSPNet gốc; dữ liệu hiện tại chưa cho thấy lợi thế rõ của 1237 khi có link hoặc atrous. CI chứa 0 không chứng minh hai phương pháp tương đương. Bootstrap ảnh chỉ đo bất định do mẫu validation, không đo biến thiên giữa các lần train.

Nhóm Kaggle dùng ResNet-50 standard stem, crop 384, micro-batch 2 × accumulation 8, freeze running statistics của mọi BN, FP16 train và FP32 eval. Cả ba đã có output cuối ở 10.000 updates/2.000 ảnh, seed 17:

| Mô hình Kaggle | mIoU (%) | Pixel accuracy (%) | Median forward FP32 trên T4 (ms) |
|---|---:|---:|---:|
| FCN–R50 | 22,9235 | 68,5614 | 53,98 |
| DeepLabV3–R50 | 20,0324 | 69,4664 | 81,22 |
| PSPNet–R50, 1236 | 25,3273 | 71,3310 | 69,37 |

Không gộp nhóm này với Modal để kết luận tác dụng kiến trúc: backbone, crop, BN, LR, precision và GPU khác nhau. DeepLabV3 hiện kém FCN trong giao thức này; điều đó không đủ kết luận DeepLabV3 nói chung kém FCN, cũng không tự chứng minh code có lỗi.

## 2. Những việc nên làm trước, không cần train lại

### 2.1. Bootstrap đúng câu hỏi về cải tiến

Ưu tiên cao nhất: dùng thống kê từng ảnh đã lưu để so sánh trực tiếp:

- b6_link − b6; b7_link − b7.
- b6_aspp − b6; b7_aspp − b7.
- Trên Kaggle: PSPNet − FCN, PSPNet − DeepLabV3.

Lấy mẫu có hoàn lại cùng danh sách ảnh cho hai mô hình; cộng intersection/GT/pred rồi tính mIoU toàn tập cho từng lần bootstrap. Không lấy trung bình mIoU từng ảnh để thay thế mIoU dataset.

Notebook mở rộng đổi tên `per_image_stats.npz` thành `per_image_stats_extension.npz`. Cell phân tích mở rộng chỉ bootstrap b7_extension − b6_extension. Vì vậy cần một bước phân tích chung đọc đúng cả hai tên và đúng output v4/v5. Hàm summarize có mã cho một số cặp link, nhưng output hiện có chưa trình bày CI trực tiếp cho tác dụng thêm link; việc đổi tên file cũng khiến vòng tìm file tên cũ bỏ qua chúng.

Đồng thời tính tương tác: (b7_link − b6_link) − (b7 − b6), và tương tự cho atrous, bằng cùng một bộ chỉ số ảnh bootstrap cho cả bốn mô hình. Kết quả hiện có gợi ý các thành phần không cộng lợi ích độc lập; cần kiểm tra thay vì chỉ so cặp 6/7 riêng lẻ.

### 2.2. Kiểm tra ảnh trùng train/validation

Output Kaggle ghi nhận 9 ảnh validation trùng byte ảnh train. Giữ kết quả trên split chính thức làm bảng chính; tính thêm mIoU và Δ trên các ảnh validation còn lại sau khi loại danh sách trùng. Nếu danh sách có đúng 9 ảnh khác nhau thì còn 1.991 ảnh. Có thể làm từ thống kê từng ảnh, không cần train lại. Dùng manifest của từng nhóm để xác nhận danh sách, không mặc định từ tên file.

### 2.3. Phân tích cải thiện và thất bại của bản mở rộng

PSPNet gốc đã có IoU từng lớp, ảnh tốt lên/xấu đi/ngẫu nhiên và đường cong train. Bổ sung cùng cách trình bày cho link/atrous so với bản gốc cùng bins, kèm số pixel và số ảnh hỗ trợ mỗi lớp.

Nếu báo cáo nhận định cải thiện vật thể nhỏ hoặc đường biên, cần đo theo diện tích thành phần liên thông của GT và dùng metric biên, với ngưỡng/độ rộng biên quy định trước và xử lý ignore nhất quán. Không suy ra kích thước vật thể từ tổng số pixel của cả lớp. Với semantic masks, thành phần liên thông chỉ là xấp xỉ vùng vật thể, không phải instance annotation.

### 2.4. Kiểm tra tính toàn vẹn giao thức

- FCN và DeepLabV3 nhúng cùng runner; runner PSPNet thêm kiến trúc và bins.
- Cả ba notebook Kaggle đều vá `lab.update` bên ngoài runner để bỏ micro-batch toàn ignore rồi trung bình gradient trên số micro-batch còn lại. Không phải lỗi chỉ xuất hiện ở một mô hình. Tuy nhiên source hash trong lock không bao phủ phần vá, và tại update đó số ảnh thực sự đóng góp gradient giảm. Cần ghi chính sách, số lần xảy ra và bước áp dụng vào báo cáo; kiểm tra lịch sử checkpoint ban đầu nếu muốn khẳng định mọi run cùng chính sách từ đầu. Không có căn cứ từ output hiện tại để buộc train lại toàn bộ.
- Với các lần chạy mới, đưa chính sách all-ignore vào runner/config được hash và đồng bộ cả ba notebook.
- Init hash của nhóm atrous v5 khác nhóm gốc/link v4. File init chứa cả lock hash nên khác hash file chưa chứng minh trọng số khác. Cần so sánh tensor chung theo tên, dtype, shape và nội dung; với classifier mở rộng so phần kênh cũ và kiểm tra phần kênh mới khởi tạo 0. Không bỏ kiểm tra lock chỉ để ghép số liệu.
- Một số markdown còn nói chưa có kết quả dù đã có output. Cần cập nhật lời dẫn trước khi nộp. Cờ kiểm tra hoàn thành hiện gắn với một seed là kiểm tra kỹ thuật của notebook, không phải tiêu chuẩn chứng minh độ tin cậy khoa học.

## 3. Hướng phát triển tương lai — không thực hiện trong phạm vi hiện tại

### 3.1. Thêm seed cho kết luận chính — ưu tiên cao

Nếu kết luận chính là đổi 6→7 cải thiện PSPNet, chạy thêm b6 và b7 với hai seed mới đã chọn trước, ví dụ 42 và 2026. Tổng cộng có ba seed; cần thêm bốn run. Trong mỗi seed, giữ cùng init phần chung, dữ liệu/augmentation và ngân sách giữa hai bản.

Báo cáo mIoU từng seed, mean ± SD và Δ ghép cặp từng seed. Ba seed là mức thực dụng ban đầu, chưa phải mẫu lớn để đưa ra khẳng định thống kê mạnh. Không chọn seed theo kết quả thuận lợi. Theo training_seconds hiện có, bốn run tương đương khoảng 7,5 giờ train trên L40S; đây là ước tính, chưa gồm validation, checkpoint, tải dữ liệu và benchmark.

Nếu khẳng định về link/atrous cũng là đóng góp chính, mở rộng seed cho chính các biến thể đó. Chạy cả sáu biến thể trên hai seed mới là 12 run bổ sung; không cần làm nếu báo cáo chỉ xem chúng là khảo sát phụ có giới hạn một seed.

### 3.2. Đối chứng dung lượng cho atrous — ưu tiên cao nếu tuyên bố hiệu quả của ASPP

Atrous trong notebook là PPM giữ nguyên cộng ba nhánh conv 3×3 dilation 12/24/36, mỗi nhánh 256 kênh; không phải thay PPM bằng toàn bộ head DeepLabV3. Số tham số train tăng từ 68,173 triệu lên 85,869 triệu, khoảng 26%; Conv/Linear MACs tăng từ khoảng 230,06 lên 293,76 G, khoảng 28%.

Để tách lợi ích của dilation khỏi dung lượng tăng, dùng đối chứng có cùng ba nhánh/cùng kênh/cùng classifier nhưng dilation [1,1,1], cùng quy tắc init và training. So với [12,24,36] ở cùng bins, trước hết 1236. Nếu chỉ có thể chạy một đối chứng mới, đây là lựa chọn có giá trị hơn việc thêm một mô hình lớn khác không cùng giao thức.

Có thể tiếp tục so dilation [6,12,18] với [12,24,36] nếu mục tiêu là lựa chọn receptive field. Đây là khảo sát thêm, không bắt buộc cho mọi báo cáo.

### 3.3. Kiểm tra ngân sách huấn luyện — ưu tiên vừa/cao tùy phạm vi kết luận

10.000 × 16 tương đương khoảng 7,9 lượt dữ liệu. Chưa đủ căn cứ để gọi kết quả là năng lực sau hội tụ. Chạy một cặp đối chứng chính, ví dụ b6/b7, ở ngân sách dài hơn như 30.000 updates, giữ mọi yếu tố còn lại cố định và thiết kế lịch LR cho toàn ngân sách ngay từ đầu.

Không sửa `max_steps` của run đã khóa hoặc gọi việc resume checkpoint cuối có LR gần 0 là thí nghiệm 30k tương đương. Dùng thí nghiệm mới; nếu cố ý fine-tune từ checkpoint 10k thì phải mô tả đó là một giao thức khác. Không nhất thiết chạy tới ngân sách paper nếu đề tài chỉ kết luận trong điều kiện giới hạn tài nguyên.

### 3.4. Thực nghiệm phụ, chỉ làm khi cần hỗ trợ nhận định cụ thể

- Nếu gọi 1237 là lựa chọn tối ưu: thử ít nhất một mức lân cận như 1235 hoặc 1238, chọn trên tập phát triển riêng và xác nhận độc lập. So 1236/1237 chỉ chứng minh khác biệt giữa hai lựa chọn đã xét.
- Nếu nhận định link học được là nguyên nhân cải thiện: đánh giá checkpoint với gates đặt 0, sau đó ablation chiều truyền một chiều/hai chiều nếu cần train thêm. Tắt gates lúc inference là phép can thiệp chẩn đoán, có thể gây lệch phân phối; không thay thế đối chứng train không có link. Độ lớn gate hoặc norm trọng số classifier chưa tự chứng minh mức đóng góp nhân quả.
- Nếu cần kiểm tra 6→7 trên backbone thứ hai: chạy thêm PSPNet–R50 1237 cùng giao thức Kaggle và init chung với 1236. Đây là kiểm tra khả năng chuyển sang giao thức/backbone khác; hai nhóm vẫn khác BN/crop/LR nên không tách riêng tác dụng backbone.
- Nếu muốn so năng lực tốt nhất của FCN/DeepLabV3/PSPNet: cần ngân sách tìm LR tương đương trên tập phát triển riêng. Bảng chung một LR hiện tại là so sánh dưới cùng giao thức giới hạn, không phải so sau tối ưu riêng cho từng mô hình.
- Nếu có tuyên bố triển khai thời gian thực: đo thêm latency toàn pipeline cho Kaggle; latency hiện tại chỉ là forward GPU trên crop. Modal đã có pipeline latency. So sánh trong cùng GPU, resolution, precision và phạm vi đo.

## 4. Kế hoạch phân tích dùng kết quả hiện có

1. Dùng hai bảng kết quả riêng: FCN/DeepLabV3/PSPNet trên Kaggle và sáu biến thể PSPNet trên Modal. Ghi rõ giao thức và số seed của mỗi bảng.
2. Từ số đã lưu, tính chênh lệch cùng bins: link−gốc và atrous−gốc. Trình bày cả tăng và giảm, không chỉ chọn trường hợp có lợi.
3. Dùng ba CI bootstrap đã có để bàn về thay đổi 6→7 ở từng nhóm. Không tự gán CI đó cho so sánh thêm link/atrous và không tạo SD giữa seed.
4. So sánh mIoU với tham số, MACs và latency trong cùng nhóm. Gọi đây là đánh đổi độ chính xác/chi phí trong cấu hình đã đo.
5. Dùng đồ thị, bảng IoU từng lớp và ảnh minh họa đã nhúng để phân tích xu hướng hội tụ và lỗi. Chỉ nêu nhận định mà dữ liệu/hình ảnh hiện có hỗ trợ; không khẳng định vật thể nhỏ hoặc biên tốt hơn nếu chưa có phép đo tương ứng.
6. Nếu CSV/NPZ gốc đã có sẵn từ các phiên cũ, có thể tính bootstrap bổ sung và bỏ ảnh trùng trên CPU. Đây là phân tích lại dữ liệu, không cần train hoặc inference mới. Hiện workspace chưa có các file này; nếu không lấy được, báo cáo vẫn dùng kết quả notebook với giới hạn đã nêu.
7. Chuyển nhiều seed, ngân sách dài hơn, đối chứng dung lượng và tập dữ liệu khác sang phần hạn chế/hướng phát triển; không coi chúng là công việc còn phải chạy.

## 5. Các nhận xét có thể đưa vào báo cáo ngay

**Ảnh hưởng của bins PPM.** Trong giao thức Modal, seed 17 và ngân sách 10.000 updates, PPM 1237 đạt 35,79% mIoU so với 34,45% của PPM 1236, tăng 1,34 điểm phần trăm. CI bootstrap theo ảnh validation là [+0,77; +1,86] điểm, cho thấy khác biệt dương trong phép phân tích lấy mẫu lại tập ảnh này. Kết quả chưa đo biến thiên giữa các lần huấn luyện.

**Ảnh hưởng của link.** Với bins 1236, thêm link tăng từ 34,45% lên 35,06%; với bins 1237, điểm giảm từ 35,79% xuống 35,17%. Vì vậy, kết quả hiện tại không hỗ trợ nhận định link luôn cải thiện PSPNet. Các gate đã học có giá trị khác 0 cho thấy cơ chế liên kết được tối ưu, nhưng không tự chứng minh nó đem lại lợi ích về chất lượng phân đoạn.

**Ảnh hưởng của atrous.** PPM 1236 kết hợp atrous đạt mIoU cao nhất trong sáu biến thể, 36,28%, cao hơn bản gốc cùng bins khoảng 1,82 điểm. Với bins 1237, mức tăng là khoảng 0,44 điểm. Đổi lại, tham số tăng khoảng 26% và MACs tăng khoảng 28%. So với b7 gốc, b6_aspp chỉ cao hơn khoảng 0,49 điểm. Do chưa có đối chứng cùng dung lượng và CI trực tiếp cho các cặp này, nên mô tả kết quả là lợi ích quan sát được của toàn bộ cấu hình mở rộng, chưa quy riêng cho dilation. Cần giữ lưu ý kiểm tra init v4/v5 đã nêu ở mục 2.4.

**Tương tác giữa các thay đổi.** Lợi thế 1237 so với 1236 giảm từ +1,34 điểm ở bản gốc xuống +0,11 điểm khi có link và -0,05 điểm khi có atrous. Hai CI ở nhóm mở rộng đều chứa 0. Điều này gợi ý hiệu quả của thay đổi bins phụ thuộc vào cấu hình đi kèm; không nên cộng các mức cải thiện riêng lẻ để dự đoán cấu hình kết hợp. Đây là nhận xét mô tả; chưa có CI trực tiếp cho tương tác.

**So sánh Kaggle.** Trong cùng giao thức Kaggle, PSPNet đạt 25,33% mIoU, cao hơn FCN khoảng 2,40 điểm và DeepLabV3 khoảng 5,29 điểm. Tuy nhiên, FCN có latency forward thấp nhất trong lần đo đã lưu: khoảng 53,98 ms, so với 69,37 ms của PSPNet và 81,22 ms của DeepLabV3. DeepLabV3 có pixel accuracy cao hơn FCN nhưng mIoU thấp hơn: hai chỉ số phản ánh chất lượng theo cách khác nhau, và pixel accuracy không thay thế mIoU. Chưa có phân tích từng lớp để quy nguyên nhân cho một nhóm lớp cụ thể.

**Giới hạn phạm vi kết luận.** Đây là khảo sát một seed, ngân sách huấn luyện giới hạn, trên validation ADE20K. Có thể báo cáo các kết quả và đánh đổi đã quan sát được, nhưng chưa kết luận cấu hình tối ưu nói chung, tái lập điểm số paper, hoặc ưu thế ổn định qua nhiều lần train. Hạn chế tài nguyên là lý do hợp lý để giới hạn phạm vi nghiên cứu; cần trình bày phạm vi đó rõ ràng.

Kết luận được dữ liệu hiện tại hỗ trợ: ở seed 17 và giao thức Modal 10k, PPM 1237 cao hơn 1236 khoảng 1,34 điểm mIoU; thêm link không cải thiện nhất quán theo bins; atrous có mIoU cao nhưng tốn thêm dung lượng và tính toán, cần CI so bản gốc và đối chứng dung lượng trước khi quy lợi ích cho cơ chế dilation. Các kết quả đều nằm trên validation đã dùng để khảo sát; nếu tiếp tục chọn cấu hình nhiều lần trên tập này, cần ghi rõ tính chất khám phá hoặc có tập kiểm tra độc lập cho kết luận cuối.

Tài liệu gốc đối chiếu: [PSPNet](https://arxiv.org/abs/1612.01105), [DeepLabV3](https://arxiv.org/abs/1706.05587). Các đề xuất ưu tiên và thiết kế thực nghiệm ở trên là đánh giá từ mã/kết quả của dự án, không phải các yêu cầu bắt buộc của hai bài báo.
