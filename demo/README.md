# PSP Studio — demo CPU trên máy local

Model: PSPNet ResNet-101 deep stem, PPM `{1,2,3,7}`, ADE20K 150 lớp; checkpoint b7 seed 17, 10.000 updates. mIoU validation đã lưu là 35,79%. Demo không cần train lại, GPU hoặc tài khoản Modal khi suy luận. Modal chỉ dùng để tải checkpoint lần đầu.

## Mở demo

1. Chạy `RUN_DEMO.cmd` ở thư mục gốc dự án.
2. Mở **http://127.0.0.1:8765** trong trình duyệt.
3. Chọn/thả một ảnh, giữ mức **512 px** rồi bấm **Bắt đầu phân đoạn**.
4. Giữ cửa sổ terminal mở. Nhấn Ctrl+C để dừng máy chủ.

Các script chạy và tải model ưu tiên `.python-runtime/python.exe` nếu có, nếu không dùng `.venv/Scripts/python.exe`. Trên máy hiện tại, runtime đi kèm dùng thư viện trong `.venv/Lib/site-packages`, nên cần giữ cả hai thư mục. File `python-runtime.zip` chỉ là gói cài đã giải nén, không cần để chạy demo.

Có ảnh mẫu phòng khách tại `demo/examples/living_room.jpg`, lấy từ ảnh demo `ADE_val_00001515.jpg` của repository hszhao/semseg.

Các thư viện, kiến trúc và checkpoint đã chuẩn bị xong thì demo có thể chạy offline. Giao diện không tải font, script hoặc tài nguyên từ Internet.

## Bạn sẽ quan sát được gì?

- Ảnh gốc và cửa sổ 473×473 mà model đang xử lý.
- Bản đồ cường độ kích hoạt thật, lấy bằng forward hooks sau stem, các tầng ResNet 1–4 và PPM. Mỗi bản đồ được chuẩn hóa riêng theo percentile 2–98 để nhìn rõ; màu giữa hai tầng không phải thước đo tuyệt đối có thể so trực tiếp. Đây không phải attention, saliency hoặc nhãn phân đoạn.
- Kết quả tạm được cập nhật sau mỗi cửa sổ. Vùng chưa xử lý giữ ảnh gốc.
- Kết quả cuối: ảnh lớp phủ, bản đồ nhãn màu, thanh kéo so với ảnh gốc và diện tích từng nhãn dự đoán.
- Tải ảnh lớp phủ, nhãn màu hoặc PNG mã lớp 0–149. Mã này đã loại label ignore gốc ADE20K; không tự cộng 1 nếu chưa kiểm tra định dạng đích.

Tiến độ cập nhật ở các mốc công việc thật, không phải thời gian còn lại ước lượng. Một tầng ResNet có thể mất lâu hơn các tầng khác. Không chèn thời gian chờ nhân tạo. Lần đầu có thêm thời gian nạp model vào RAM; các lần sau dùng lại model.

## Chế độ suy luận

- **512 px**: resize cạnh dài 512, sliding windows 473, stride 316, FP32, không flip. Trung bình xác suất ở vùng chồng lấn, nội suy xác suất rồi mới argmax, giống giao thức notebook b7.
- **384 px**: nhanh hơn, thường một cửa sổ; thay đổi kích thước đánh giá nên không gán điểm mIoU 35,79% cho chế độ này.
- **768 px**: thường có nhiều cửa sổ hơn, mất nhiều thời gian hơn; không mặc định chất lượng tốt hơn.

Ảnh đầu vào được xoay đúng EXIF và thu nhỏ cạnh dài tối đa 1.600 px để hạn chế RAM khi xuất kết quả. Ảnh tối đa 20 MB và 16 megapixel. Điều này có thể làm khác đầu ra so với đánh giá ảnh gốc lớn trong notebook. Phần tiền xử lý cố định của notebook giữ nguyên ở mức 512 cho ảnh không bị thu nhỏ/đổi hướng.

## Cấu trúc

```text
checkpoints/b7_seed17_last.pt   # checkpoint gốc, khoảng 521 MiB
demo/inference.py              # load CPU và suy luận có tiến độ
demo/server.py                 # HTTP local, streaming NDJSON
demo/web/                      # giao diện không cần build
demo/vendor/semseg/             # mã MIT của hszhao/semseg, pinned commit
```

Mã upstream được cố định tại commit `4f274c3f276778228bc14a4565822d46359f0cc8`. Demo định nghĩa forward trả logits giống notebook và nạp model bằng `strict=True`; không dùng torchvision ResNet thay thế. Checkpoint được đọc bằng `weights_only=True`, `map_location="cpu"`, `mmap=True`; không cần nạp optimizer vào model.

## Cài lại nếu chuyển máy

Nếu clone repository chính bằng `--recurse-submodules`, mã `demo/vendor/semseg` đã có sẵn; bỏ qua hai lệnh clone/checkout semseg bên dưới. Checkpoint tải bằng Git LFS cũng dùng được trực tiếp, không cần tải lại từ Modal.

Yêu cầu Python 3.12 x64 và Git. Tại thư mục dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r demo/requirements.txt
git clone https://github.com/hszhao/semseg.git demo/vendor/semseg
git -C demo/vendor/semseg checkout --detach 4f274c3f276778228bc14a4565822d46359f0cc8
```

Nếu chưa có checkpoint, đăng nhập Modal bằng `.\.venv\Scripts\python.exe -m modal setup`, sau đó chạy `DOWNLOAD_MODEL.cmd`. Script tải từ Volume **pspnet-study-ade20k**, environment **main**, đường dẫn `results/ade20k_seed17_b16_10k_v4/b7_seed17/last.pt`; kiểm tra SHA256 trùng output notebook trước khi dùng. Máy hiện tại có model gốc ở profile **duonglehieuhy**; profile **lethidieuthao2007** chỉ có các bản link trong thư mục v4. Script chỉ chọn profile cho lệnh tải, không đổi profile mặc định. Nếu chuyển tài khoản/máy, chỉnh tên profile trong script theo tài khoản sở hữu checkpoint. Không cung cấp token cho giao diện demo.

Để đổi cổng hoặc số luồng CPU:

```powershell
.\.venv\Scripts\python.exe demo/server.py --port 8766 --threads 4
```

Ứng dụng chỉ lắng nghe `127.0.0.1`, xử lý một ảnh tại một thời điểm và không lưu ảnh upload xuống đĩa. Tốc độ tùy CPU; nên thử ảnh tĩnh trước buổi demo. Không có nhãn ground truth thì không tính được mIoU cho ảnh vừa tải lên.

## Kiểm tra đã thực hiện trên máy này

- Checkpoint 546.162.205 bytes, SHA256 khớp output notebook: `b7f060e54ec271ea41049daa18f82d607b2581b93d642884971d192650ac89cd`.
- Chạy CPU với ảnh ADE demo, mức 512 px và 2 cửa sổ: khoảng 7,3–7,9 giây trong hai lần kiểm tra, gồm nạp model lần đầu của mỗi tiến trình. Không coi đây là benchmark đại diện cho mọi ảnh hoặc máy.
- Đầu ra mã lớp khớp từng pixel với hàm tham chiếu theo notebook trên ảnh kiểm tra; streaming có đủ 6 mốc tầng cho mỗi cửa sổ và các hooks được gỡ sau forward.
- Chrome thực: upload, suy luận trực tiếp, lớp phủ/nhãn màu, thanh so sánh và dữ liệu PNG tải xuống đều qua kiểm tra; viewport desktop 1440×1100 và mobile 390×844 không có JavaScript exception hoặc tràn ngang trên mobile.
- Báo cáo tại `demo/validation/`. Ảnh `desktop-result.png` được giữ làm nguồn dựng slide; ảnh kiểm thử khác và log có thể xóa rồi tạo lại bằng các lệnh bên dưới. Chạy `RUN_DEMO.cmd` để bắt đầu phiên demo của bạn.

Chạy lại kiểm tra khi sửa phần suy luận:

```powershell
.\.venv\Scripts\python.exe -X utf8 demo/validate.py
node demo/browser_check.mjs
```

Trên máy hiện tại, thay lệnh Python bên trên bằng `.\.python-runtime\python.exe -X utf8 demo/validate.py` vì Python gốc mà `.venv` tham chiếu không còn tồn tại. Script kiểm tra trình duyệt tự chọn runtime giống `RUN_DEMO.cmd`.

Kiểm tra trình duyệt dùng Chrome cài tại đường dẫn mặc định trên Windows và Node.js 24, không yêu cầu cài Playwright. Danh sách phiên bản thư viện thực tế nằm trong `demo/requirements-resolved.txt`.
