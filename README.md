# PSPNet trên Modal — ADE20K, một seed

Demo CPU trên máy hiện tại: chạy `RUN_DEMO.cmd`, xem [demo/README.md](demo/README.md). Nhật ký dọn file thừa ngày 02/11/2026: [luu_tru/NHAT_KY_DON_DEP_02_11_2026.json](luu_tru/NHAT_KY_DON_DEP_02_11_2026.json).

Notebook đối chứng **FCN và DeepLabV3 trên Kaggle**: xem [README_KAGGLE.md](README_KAGGLE.md),
[fcn-kaggle-ade20k.ipynb](fcn-kaggle-ade20k.ipynb) và [deeplabv3-kaggle-ade20k.ipynb](deeplabv3-kaggle-ade20k.ipynb).
Đối chứng **PSPNet cùng cấu hình Kaggle**: [pspnet-kaggle-ade20k.ipynb](pspnet-kaggle-ade20k.ipynb).

So sánh PPM `{1,2,3,6}` và `{1,2,3,7}` bằng ADE20K thật với **seed 17**.
Mỗi bản huấn luyện 10.000 optimizer updates với batch 16 (tổng 20.000) (paper: 150.000), cùng init, dữ liệu/augmentation và ngân sách.

| File | Vai trò |
|---|---|
| [pspnet-modal.ipynb](pspnet-modal.ipynb) | Notebook chính, tự chứa mã để upload lên Modal |
| [modal_app.py](modal_app.py) | GPU worker chạy bằng Modal CLI |
| [pspnet_lab.py](pspnet_lab.py) | Tải tài nguyên, mô hình, train, đánh giá và benchmark |
| [configs/modal_research.json](configs/modal_research.json) | Cấu hình CLI cho ADE20K, một seed |

## Chạy notebook

1. Upload notebook vào https://modal.com/notebooks.
2. Chọn **GPU L4** (hoặc L40S/A100, giữ cố định cả thí nghiệm), khoảng **4 CPU cores**, **16 GiB RAM** trong Compute profile.
3. Tạo **Volume v2** (ví dụ `pspnet-study-ade20k-v2`), attach tại `/mnt/pspnet-study-ade20k` hoặc `/mnt/pspnet-study-ade20k-v2`; notebook tự nhận đường dẫn đang gắn.
4. Chạy các cell tuần tự. Mặc định đã là `research`, ADE20K 150 lớp, `seeds=[17]`.
5. Notebook tự tải ADEChallengeData2016 và pretrained `resnet101_v2.pth` của tác giả nếu chưa có.
6. Self-test và **preflight với dữ liệu thật** chạy trước train dài: kiểm tra weights, loss/gradient, forward, số tham số và peak VRAM.
7. Preflight update bị bỏ đi; cả hai bản train từ init chung. Các cell sau đánh giá, benchmark và xuất bảng/ảnh.

Có thể tạo Volume v2 bằng CLI rồi attach trong Files/Volumes:

```powershell
python -m pip install "modal>=1.6.1,<2"
python -m modal setup
python -m modal volume create pspnet-study-ade20k-v2 --version=2
```

Volume v2 được đồng bộ bằng `sync -f`, tránh reload khi log vẫn mở trong notebook/subprocess.
Không dùng Volume v1 cũ cho cấu hình này. Mở lại cùng Volume, GPU và môi trường để resume checkpoint.

## Dữ liệu và weights

Research kiểm tra 20.210 ảnh train, 2.000 ảnh validation. Raw label 0 là ignore, 1..150 chuyển thành 0..149.
Split chính thức có vài ảnh validation trùng byte với ảnh train; giữ nguyên split và ghi vào `train_val_duplicates.json`.
Weights tải từ [file chính thức của tác giả](https://drive.google.com/file/d/1V-sfLnqSuwTPgNMrqUp4HDZpqtoBH4xm/view).
Nếu Google Drive giới hạn tải, tải thủ công rồi upload vào:

`/mnt/pspnet-study-ade20k-v2/weights/resnet101_v2.pth`

Weights được kiểm tra keys/shapes trước khi tải dataset lớn. Không thay bằng torchvision ResNet-101 vì stem khác.
Nếu dùng checkpoint PSPNet, đổi `INIT_KIND=psp_checkpoint` và đặt đường dẫn; cả hai bản cùng train tiếp với optimizer reset.

## Cấu hình

- Một seed 17, hai bản b6/b7; ResNet-101, dilation, OS8, số kênh, heads và auxiliary loss giữ như nhau.
- Crop 473; batch thực 16, không accumulation (BatchNorm thấy 16 ảnh như paper).
- Train AMP **BF16** trên L4; đánh giá/benchmark **FP32**, TF32 tắt.
- SGD, backbone LR 0.01, head LR ×10, momentum 0.9, weight decay 1e-4, poly power 0.9.
- 10.000 updates mỗi bản (~8 epoch), checkpoint bước cuối; toàn validation, single-scale, không flip.
- Log từng bước (`train.csv`), đánh giá 500 ảnh validation cố định mỗi 2.000 bước (`val_history.csv`, chỉ để theo dõi),
  snapshot checkpoint mỗi 2.000 bước, và paired bootstrap cho Δ mIoU (`paired_bootstrap.csv`).

`preflight.json` ghi VRAM và ước lượng thời gian từ một update (có cold-start).
Nếu OOM, chọn GPU lớn hơn hoặc thay crop/batch cho cả hai bản trong thí nghiệm mới.
Không đổi protocol giữa hai bản. Resume khôi phục model, optimizer, RNG và vị trí dữ liệu, không reset LR.

## Chạy CLI thay cho notebook

Dùng output riêng cho CLI và notebook. Không chạy đồng thời vào cùng thư mục kết quả.

```powershell
python -m modal run modal_app.py --action prepare
python -m modal run modal_app.py --action preflight
python -m modal run --detach modal_app.py --action train --variant b6 --seed 17 --chunk-steps 1000
```

Mỗi lệnh train thêm tối đa 1.000 updates. Chạy lại tới `complete=True`; thực hiện tương tự cho b7.
`--detach` chỉ giữ đoạn đang gửi chạy, không tự gửi đoạn tiếp theo. Mỗi Function có timeout 24 giờ.

Sau khi mỗi bản đủ ngân sách:

```powershell
python -m modal run modal_app.py --action evaluate --variant b6 --seed 17
python -m modal run modal_app.py --action evaluate --variant b7 --seed 17
python -m modal run modal_app.py --action benchmark --variant b6 --seed 17 --repeat 0
python -m modal run modal_app.py --action benchmark --variant b7 --seed 17 --repeat 0
```

Benchmark repeat từ 0 đến 4, luân phiên thứ tự hai bản; sau đó:

```powershell
python -m modal run modal_app.py --action summarize
python -m modal volume get pspnet-study-ade20k-v2 /results/ade20k_seed17_b16_10k_v4 ./modal_results
```

## Diễn giải kết quả

Xuất mIoU, IoU từng lớp, confusion matrix, tham số, MACs/FLOPs Conv/Linear,
fvcore với unsupported operators, latency median/p95, VRAM và ảnh tăng/giảm chất lượng.
Một seed không cung cấp SD giữa các lần huấn luyện; bảng không bịa SD hoặc ý nghĩa thống kê.
Các phiên benchmark lặp lại vẫn được giữ để đo latency ổn định hơn.

BatchNorm batch thực 2 không tương đương SyncBatchNorm batch 16 của giao thức bài gốc.
Đây là so sánh hai cấu hình trong điều kiện đã nêu, không khẳng định tái tạo chính xác điểm số paper.
Không có số liệu nghiên cứu điền sẵn hoặc khẳng định 7×7 chắc chắn tốt hơn.

Tài liệu: [paper](https://arxiv.org/abs/1612.01105), [mã tác giả](https://github.com/hszhao/semseg),
[Modal Notebooks](https://modal.com/docs/guide/notebooks), [Volume v2](https://modal.com/docs/guide/volumes).

Rà soát triển khai: tải có kiểm tra độ dài và retry; preflight tách bộ nhớ train/inference; manifest được khóa trước khi ghi đè. Output mặc định v4 (10k bước mỗi bản, batch 16) tránh trộn checkpoint từ mã cũ.

## Lấy toàn bộ dự án từ GitHub

Cài Git và Git LFS, sau đó chạy:

```powershell
git lfs install
git clone --recurse-submodules https://github.com/LDH-LeHieu18012005/scene-semantic-segmentation.git
cd scene-semantic-segmentation
git lfs pull
```

Checkpoint `checkpoints/b7_seed17_last.pt` được lưu bằng Git LFS. Mã hszhao/semseg được ghim bằng submodule; nếu đã clone thiếu submodule, chạy `git submodule update --init --recursive`. Môi trường `.venv` và `.python-runtime` là các cài đặt riêng của máy, không đưa vào Git; xem `demo/README.md` để cài thư viện và chạy demo sau khi clone. Nếu đã có checkpoint từ Git LFS thì không cần chạy `DOWNLOAD_MODEL.cmd`.
