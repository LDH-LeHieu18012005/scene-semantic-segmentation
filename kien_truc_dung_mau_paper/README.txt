BỘ KIẾN TRÚC THEO ĐỊNH DẠNG HÌNH PSPNET GỐC

Mỗi cấu hình có PNG 5100 px chiều ngang và SVG chỉnh sửa được:
b6, b7, b6_link, b7_link, b6_aspp, b7_aspp.

Yêu cầu thiết kế: giữ luồng ngang Input Image → CNN → Feature Map → POOL → bốn nhánh PPM → UPSAMPLE → CONCAT → CONV → Final Prediction; nền trắng, nét mảnh, chữ Times New Roman, khối dây có màu đỏ/cam/xanh lam/xanh lục và khung chấm như hình tham chiếu. Không dùng các ô pastel hoặc thay toàn bộ bằng sơ đồ khối.
- b6/b7: chỉ khác lưới pooling cuối 6×6 hoặc 7×7.
- Link: đúng sáu cạnh có hướng từ đặc trưng ban đầu của các nhánh kề nhau đến phép cộng tại nhánh đích; resize/pool × tanh(w), trước UPSAMPLE.
- Atrous: giữ PPM, bổ sung ba nhánh Conv 3×3 dilation 12/24/36 nhận F trực tiếp; 256 kênh/nhánh, ghép tại CONCAT thành 4864 kênh.

Nguồn hình tham chiếu: Zhao và cộng sự, Pyramid Scene Parsing Network, CVPR 2017; ảnh nguồn demo/vendor/semseg/figure/pspnet.png từ mã tham chiếu hszhao/semseg. Hai ảnh ở đầu/cuối sơ đồ được lấy từ hình tham chiếu để minh họa luồng kiến trúc; không phải dự đoán mới của từng biến thể. Các phần thay đổi được đối chiếu pspnet_lab.py.

Đã thử chỉnh sửa bằng công cụ imagegen tích hợp với ràng buộc giữ hình tham chiếu, nhưng bản thử làm mất một số nét kỹ thuật; không đưa các bản thử đó vào tài liệu. Bộ hình bàn giao được dựng lại thành SVG kỹ thuật và PNG bằng script luu_tru/nguon_hinh_kien_truc/paper_format_figures.ps1 để kiểm soát chính xác hình học và số cạnh. Không dùng API/CLI fallback.

Word: thay hình tổng thể cũ và trình bày đủ sáu cấu hình theo mẫu. Giữ phần giải thích link từng bước và MACs.
PowerPoint: 18 slide; chữ trình chiếu Arial theo cấp nội dung, tiêu đề học thuật. Slide 14 bổ sung giao diện PSP Studio từ demo/validation/desktop-result.png; các hình giải thích link sử dụng câu chữ học thuật.

Kiểm tra: xem PNG, phân tích 168 phần XML và 239 liên kết nội bộ; hai hình link có đúng sáu cạnh có hướng; gói Office dùng tên mục ZIP theo chuẩn dấu /. Chưa render phân trang bằng Office trong phiên này. Khi mở Word, cập nhật mục lục để đồng bộ số trang.
Sau dọn dẹp: giữ PNG/SVG riêng, bỏ ZIP trùng nội dung; các hình giải thích link/MACs nằm trong giai_thich_chi_tiet/.
