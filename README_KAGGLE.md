# FCN, DeepLabV3 và PSPNet trên Kaggle

Ba notebook độc lập, tự chứa runner Python:

| Notebook | Mô hình |
|---|---|
| [fcn-kaggle-ade20k.ipynb](fcn-kaggle-ade20k.ipynb) | TorchVision FCN–ResNet-50 |
| [deeplabv3-kaggle-ade20k.ipynb](deeplabv3-kaggle-ade20k.ipynb) | TorchVision DeepLabV3–ResNet-50, không phải DeepLabV3+ |
| [pspnet-kaggle-ade20k.ipynb](pspnet-kaggle-ade20k.ipynb) | PSPNet với cùng backbone TorchVision ResNet-50, PPM 1236 mặc định |

## Chạy lần đầu

1. Upload một notebook vào Kaggle, bật GPU và Internet. Mã dùng một GPU (`cuda:0`); không gộp VRAM T4×2.
2. Khuyến nghị Add Input dataset **ADEChallengeData2016 gốc đã giải nén**. Root phải chứa:

   ```text
   ADEChallengeData2016/
     images/training/*.jpg          # 20.210 ảnh
     images/validation/*.jpg        # 2.000 ảnh
     annotations/training/*.png
     annotations/validation/*.png
   ```

   Notebook tự tìm root dưới `/kaggle/input`. Nếu có nhiều bản, đặt `DATA_ROOT` rõ ràng. Nếu chưa attach dữ liệu, mặc định tự tải ZIP chính thức của MIT vào `/kaggle/working/data`.

3. Nhãn phải còn nguyên: `0` là ignore; `1..150` là lớp gốc. Notebook tự chuyển thành `255` / `0..149`. Không dùng bộ đã remap.
4. Chạy cell môi trường, cấu hình, ghi runner, chuẩn bị dữ liệu, rồi preflight. Không cài đè torch/torchvision mặc định của Kaggle.
5. Preflight stable v2 chạy năm optimizer updates tạm trên dữ liệu thật với đủ accumulation và các batch khác nhau, kiểm tra gradient/forward và báo VRAM/ETA. Các update này bị bỏ đi. Nếu preflight thành công, chạy cell train.
6. Khi đủ 10.000 update, các cell sau đánh giá toàn validation, benchmark và xuất kết quả. Nếu hết ngân sách phiên trước đó, phần đánh giá cuối được bỏ qua và có hướng dẫn resume.

Chạy các notebook trong phiên Kaggle riêng. Mỗi notebook có output và tên thí nghiệm riêng.

## Cấu hình và tính công bằng

- Cùng ResNet-50 ImageNet-1K V1; segmentation head 150 lớp khởi tạo mới, auxiliary loss hệ số 0,4.
- Không dùng segmentation pretrained COCO/VOC/ADE20K.
- Crop 384×384; micro-batch 2; accumulation 8; batch hiệu dụng 16; seed 17; 10.000 optimizer updates.
- Stable v2: SGD momentum 0,9; backbone LR 0,001; heads LR ×10 (tối đa 0,01); weight decay 1e-4; poly power 0,9. Linear warmup 500 update từ hệ số 0,1; clip gradient norm 1,0 sau khi unscale.
- Augmentation scale 0,5–2, rotation ±10°, flip, blur; được xác định theo seed/epoch/index nên không phụ thuộc thứ tự workers hoặc kiến trúc.
- FP16 + GradScaler khi train; loss cross-entropy tính FP32. Nếu forward FP16 overflow, thử lại toàn update bằng FP32 với cùng dữ liệu/RNG, xóa gradient lần thử hỏng. Gradient overflow được thử lại với scale nhỏ hơn, rồi chuyển sang thử FP32 nếu cần. Log `amp_retries` và `fp32_fallback` ghi rõ các lần thử; không tính lần thử hỏng thành optimizer update. Đánh giá FP32, TF32 tắt, không yêu cầu BF16.
- Freeze running mean/variance của **mọi BatchNorm**; affine scale/bias vẫn học. Đây là lựa chọn cho batch thực nhỏ, không tương đương BN batch thực 16 trên Modal.
- Theo dõi 100 ảnh validation cố định mỗi 1.000 update; không chọn checkpoint tốt nhất. Báo cáo dùng checkpoint cuối.
- Đánh giá longest-side resize 512, sliding windows crop 384, không flip, cùng label mapping. mIoU từ tổng confusion matrix của toàn bộ 2.000 ảnh.
- Lock kiểm tra cấu hình, nội dung dữ liệu, source runner, torch/torchvision và GPU. Đổi giao thức/môi trường thì dùng tên thí nghiệm mới; không trộn checkpoint.

Nếu OOM: đổi micro-batch thành 1, accumulation thành 16 cho **tất cả mô hình trong bảng so sánh**, đặt tên thí nghiệm mới và preflight lại. Có thể giảm crop nếu vẫn thiếu bộ nhớ. Không tự thay đổi cấu hình sau OOM.

Nếu CPU/DataLoader worker lỗi: dùng `cfg["workers"] = 0`; thay đổi này không làm đổi thứ tự dữ liệu. Các cài đặt `session_steps`, `session_hours`, `workers`, chu kỳ log/save không làm reset lịch LR.

## Lưu và resume qua phiên Kaggle

`/kaggle/working` không phải ổ đĩa bền vững tự động. Phải Save Version/Commit và kiểm tra output đã lưu trước khi bỏ phiên; khi phiên bị hủy mà chưa có output được lưu, không thể đảm bảo cứu checkpoint.

1. Output mỗi lần chạy nằm tại `/kaggle/working/<EXPERIMENT_NAME>/`.
2. Checkpoint đầy đủ `last.pt` lưu atomic ngay sau update thành công đầu tiên và mỗi 250 update, gồm model, momentum optimizer, GradScaler, RNG, bước và khóa giao thức. Nếu lỗi số học vẫn xảy ra trong FP32, giữ checkpoint ở bước thành công gần nhất và ghi `numerical_failure.json`. Không lưu hàng chục snapshot lớn để tránh đầy dung lượng.
3. Mặc định cell train chạy tối đa 10.000 bước trong lần gọi và tự dừng/lưu tại ranh giới update sau khoảng 7 giờ. Đặt `TRAIN_STEPS_THIS_SESSION` nhỏ hơn để chủ động chia phiên. Cần chừa thời gian cho validation và lưu output; notebook không thể gia hạn quota của Kaggle.
4. Cùng session: chạy lại cell train để tiếp tục; bỏ trống `RESUME_FROM`.
5. Phiên mới: Add Input output notebook đã lưu, đặt `RESUME_FROM` tới `.../<EXPERIMENT_NAME>/last.pt`, giữ nguyên mô hình/tên/giao thức/GPU/phần mềm. Notebook sao chép output sang thư mục writable và tiếp tục. Không ghi đè checkpoint đã tiến xa hơn trong cùng session.
6. Không xóa optimizer/RNG để resume. Không sửa `max_steps` sau khi đã bắt đầu; đổi ngân sách là thí nghiệm mới.

Nếu Internet tắt, attach `resnet50-0676ba61.pth` (ImageNet V1 chính thức) rồi đặt `BACKBONE_WEIGHTS`. Nếu resume checkpoint hợp lệ, không cần tải lại backbone. Mã không tự chuyển sang random weights khi tải thất bại.

## Output

- `metrics.json`, `summary.csv`: mIoU, pixel accuracy, ngân sách và metadata.
- `per_class.csv`, `confusion.npy`: IoU từng lớp và confusion matrix.
- `per_image_stats.npz`: intersection/GT/pred counts theo ảnh, dùng cho paired bootstrap sau này.
- `train.csv`, `val_history.csv`, `training_loss.png`, `prediction_example.png`.
- `benchmark.json`: tham số train/inference, MACs Conv2d, latency median/p95, peak allocated VRAM. Nhánh aux không chạy khi benchmark. Không tính latency này là tốc độ toàn pipeline.
- `<EXPERIMENT_NAME>_report.zip`: bảng/log/ảnh, **không chứa `last.pt`**. Giữ checkpoint riêng để resume.

Không gộp trực tiếp benchmark latency T4 với L40S. Không coi so sánh ResNet-50/crop384/BN freeze trên Kaggle với ResNet-101/crop473/batch thực16 trên Modal là ablation kiến trúc. Dùng notebook PSPNet Kaggle mới làm đối chứng cùng giao thức. Một seed không đo được độ ổn định qua các lần train.

## PSPNet làm đối chứng Kaggle

`pspnet-kaggle-ade20k.ipynb` dùng đúng backbone standard-stem TorchVision ResNet-50 / OS8 và auxiliary head như FCN; thêm PPM bốn nhánh 512 kênh, head Conv3×3 4096→512 và classifier 150 lớp. Bilinear interpolation dùng `align_corners=False` giống nhóm Kaggle. Đây là bản PSPNet thích nghi cho phép so sánh chung backbone, không phải kiến trúc deep-stem ResNet-101 của nhóm Modal.

Cấu hình mặc định là `PPM_BINS=[1,2,3,6]`, tên `pspnet_r50_b6_ade20k_s17_c384_b16_10k_stable_v2`. Để thử bản 1237, đổi thành `PPM_BINS=[1,2,3,7]` và dùng tên thí nghiệm mới (ví dụ đổi `b6` thành `b7`). Bins được lưu trong khóa giao thức; không đổi giữa chừng hoặc dùng lại checkpoint khác bins.

Source train/eval của notebook này được kế thừa từ runner Kaggle, chỉ bổ sung kiến trúc và cấu hình bins. Cả ba notebook đã đồng bộ giao thức stable v2 để giữ tính công bằng sau sửa lỗi số học. Dùng tên output mới, không resume checkpoint/lock v1; các kết quả v1 đã lưu cần giữ riêng. Nếu muốn bảng so sánh cùng giao thức, chạy cả ba bằng v2.

## Sửa lỗi số học v1 → stable v2

Notebook PSPNet v1 đã gặp `FloatingPointError: Non-finite forward loss` trong train trên T4, mặc dù preflight qua và peak VRAM chỉ khoảng 2,36 GiB. Đây không phải lỗi OOM. Head LR 0,1 với BN running statistics bị freeze, không warmup/clipping, là nguyên nhân đáng nghi gây mất ổn định; log hiện có không đủ để xác định riêng từng yếu tố. Preflight cũ chỉ lặp một micro-batch nên chưa phản ánh train với accumulation và dữ liệu thay đổi.

V2 giảm LR, thêm warmup/clipping, tăng độ sát thực của preflight, xử lý FP16 forward overflow bằng FP32 retry, và lưu checkpoint sớm. Không dùng `nan_to_num`, không bỏ batch lỗi, không tiếp tục cập nhật khi FP32 vẫn không hữu hạn. Trên Kaggle, upload notebook mới, restart session/kernel, để `RESUME_FROM=""` cho lần chạy v2 đầu tiên, rồi chạy lại từ đầu. Dataset đã attach/tải có thể dùng lại; không cần xóa kết quả v1.

## Mã nguồn và kiểm tra trước khi chạy

`kaggle_segmentation.py` là bản source của FCN/DeepLabV3 để tham khảo/chỉnh sửa; PSPNet nhúng bản mở rộng riêng trong notebook. Mỗi notebook đã nhúng mã nên người chạy chỉ cần upload `.ipynb`. Nếu chỉnh sửa code dùng chung, cần đồng bộ cell ghi runner trong các notebook liên quan.

Ba mô hình đã được kiểm thử trên CPU với dữ liệu tổng hợp nhỏ: cú pháp các cell, nhãn, confusion matrix, sampler, forward/backward, checkpoint/resume, đánh giá và benchmark. PSPNet còn được kiểm tra forward 384×384 và kích thước lẻ, output stride 8, đồng nhất cấu hình/train/eval với FCN/DeepLabV3, nạp cùng backbone state và khóa bins 1236/1237. Kiểm thử CPU không xác nhận VRAM/tốc độ/khả năng chạy toàn ADE20K trên Kaggle; cell preflight thực hiện kiểm tra đó trên GPU thật trước train dài.

Regression stable v2 đã kiểm tra: cả ba mô hình train có accumulation và resume cho trọng số cuối giống chạy liên tục trên CPU; AMP forward overflow và norm overflow được tiêm có chủ đích để kiểm tra retry không làm SGD cập nhật nhầm; lỗi FP32 thật bị chặn và giữ checkpoint. Với backbone ImageNet V1 chính thức và batch tổng hợp có nhãn theo vùng, lịch LR v1 tái hiện được mất ổn định; các update v2 ở LR trong warmup và sau warmup vẫn hữu hạn, gradient sau clip không vượt ngưỡng. Đây là kiểm thử CPU/overflow mô phỏng, không phải lần train ADE20K đầy đủ trên GPU Kaggle.

Nguồn: [FCN TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html), [DeepLabV3 TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html), [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).
