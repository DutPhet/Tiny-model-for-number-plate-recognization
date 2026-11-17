# Tiny Model for Number Plate Recognition & Tracking (Edge ALPR)

Hệ thống nhận diện và theo dõi biển số xe tự động (ALPR - Automatic License Plate Recognition) siêu nhẹ (Tiny Edge ALPR), được thiết kế tối ưu hóa chuyên biệt cho **camera hành trình (dashcam)** và **thiết bị biên (Edge Devices)**.

Hệ thống kết hợp kiến trúc tiên tiến: **YOLO11n** + thuật toán theo dõi **ByteTrack** + mạng biến đổi không gian **STN (Spatial Transformer Network)** + mạng nhận diện ký tự siêu nhẹ **LPRNet** + cơ chế bỏ phiếu thời gian **Temporal Majority Voting**.

---

## 🌟 Điểm nổi bật & Tính năng cốt lõi

1. **Siêu nhẹ & Tối ưu cho Thiết bị Biên (Ultra-lightweight Edge)**:
   - Bộ dò biển số & xe: **YOLO11n** (~5.2 MB, 2.6M params).
   - Bộ đọc ký tự: **STN + LPRNet** (~4.15 MB, 1.08M params).
   - Tổng dung lượng toàn bộ pipeline chưa tới **10 MB**, sẵn sàng triển khai trên Jetson Nano, Raspberry Pi, camera AI hoặc CPU thông thường.
2. **Khắc phục rung lắc & góc nghiêng Camera hành trình**:
   - Module **STN tự động căn chỉnh hình học**: Bounded Affine Transformation cân bằng góc nghiêng ($\pm 14^\circ$), co giãn ($[0.8, 1.2]\times$) và dịch chuyển $(\pm 25\%)$ để nắn phẳng biển số trước khi OCR.
3. **Khử hoàn toàn Watermark & Nhiễu nền (Cascaded 2-Stage Detection)**:
   - Chỉ tìm kiếm biển số bên trong bounding box của các phương tiện được phát hiện (`car_crop`), loại bỏ 100% hiện tượng nhận diện nhầm logo, watermark hay timestamp ở góc video camera hành trình.
4. **Bộ giải mã CTC Loss không phân đoạn (Segmentation-free OCR)**:
   - Sử dụng CTC Loss trực tiếp trên chuỗi đặc trưng, không cần cắt rời từng chữ cái hay ký tự đơn lẻ.
5. **Khử nhòe chuyển động bằng Temporal Majority Voting**:
   - Kết hợp `track_id` từ **ByteTrack** để duy trì lịch sử nhận diện qua nhiều khung hình liên tiếp. Bỏ phiếu đa số theo tần suất ký tự giúp loại bỏ hoàn toàn các khung hình bị rung nhòe, chói sáng hoặc khuất một phần.

---

## 📁 Cấu trúc Thư mục Dự án (Project Structure)

```text
Tiny-model-for-number-plate-recognization/
├── data_source/                                # Dữ liệu hình ảnh và dataset
│   ├── Vietnam license-plate.v1i.yolov8/       # Dataset YOLOv8 huấn luyện phát hiện biển số
│   ├── dataset_ocr/                            # Dataset OCR tổng hợp (train/val)
│   ├── real_cropped_plates/                    # Biển số thực tế cắt từ dữ liệu với smart padding
│   └── organized_frames_dataset/               # Dữ liệu clip chuỗi khung hình từ camera hành trình
├── notebook/                                   # Thư mục chứa các Jupyter Notebook nghiên cứu & báo cáo
│   ├── view_boxes.ipynb                        # Trực quan hóa tracking xe & biển số
│   └── compare_lprnet_experiments.ipynb        # Báo cáo so sánh & đánh giá thực nghiệm LPRNet + STN
├── output/                                     # Kết quả gán nhãn, tracking và video đã chú thích
├── runs/                                       # Trọng số và lịch sử huấn luyện
│   ├── train/yolo11n_license_plate/weights/    # Trọng số YOLO11n tốt nhất (best.pt)
│   ├── train/lprnet_bounded_stn/weights/       # Trọng số LPRNet Bounded STN tốt nhất (best.pt)
│   └── train/lprnet_two_stage/weights/         # Trọng số LPRNet Two-Stage Warmup (best.pt)
├── auto_track_label.py                         # Module Tracking & Auto-labeling song song (Xe + Biển số)
├── crop_plates.py                              # Module cắt biển số thực tế từ ảnh xe với padding thông minh
├── data.yaml                                   # Cấu hình dữ liệu huấn luyện YOLO11n (đường dẫn tương đối)
├── generate_synth_plates.py                    # Trình sinh dữ liệu biển số Việt Nam nhân tạo (Augmented)
├── model_stn_lprnet.py                         # Định nghĩa kiến trúc PyTorch STN + LPRNet & CTC Decoder
├── pipeline_end2end.py                         # Pipeline End-to-End toàn diện (YOLO11n + ByteTrack + STN + LPRNet)
├── requirements.txt                            # Danh sách thư viện phụ thuộc hoàn chỉnh
├── train.py                                    # Huấn luyện mô hình YOLO11n phát hiện biển số xe
├── train_lprnet.py                             # Huấn luyện mạng LPRNet + STN với CTC Loss & In-Memory Cache
├── compare_lprnet_experiments.ipynb            # Notebook so sánh thực nghiệm tại thư mục gốc
├── view_boxes.ipynb                            # Notebook trực quan hóa tại thư mục gốc
├── yolo11n.pt                                  # Trọng số COCO cơ sở (phát hiện phương tiện)
└── README.md                                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cài đặt Môi trường (Installation)

Yêu cầu Python 3.10 trở lên:

```bash
# 1. Tạo môi trường ảo (khuyến nghị)
python -m venv myenv

# 2. Kích hoạt môi trường (Windows PowerShell)
.\myenv\Scripts\Activate.ps1

# Hoặc kích hoạt trên Linux / macOS:
# source myenv/bin/activate

# 3. Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

> **Lưu ý**: Dự án đã tích hợp `lapx` để hỗ trợ giải thuật gán tuyến tính (Linear Hungarian Assignment) cho bộ theo dõi **ByteTrack** trên Windows/Linux.

---

## 🚀 Quy trình Thực thi Dự án (Workflow & Execution)

### Bước 1: Huấn luyện Mô hình Dò Biển Số (YOLO11n Plate Detector)
Fine-tune mô hình nền `yolo11n.pt` trên bộ dữ liệu biển số xe Việt Nam:
```bash
python train.py
```
- **Cấu hình**: `data.yaml` tự động nhận diện đường dẫn tương đối tới `data_source/Vietnam license-plate.v1i.yolov8`.
- **Trọng số tốt nhất**: Tự động lưu tại `runs/train/yolo11n_license_plate/weights/best.pt`.

---

### Bước 2: Chuẩn bị Dữ liệu Huấn luyện OCR (Synthetic & Real Cropping)

#### 2.1. Sinh biển số Việt Nam tổng hợp (Data Augmentation)
Tạo tập dữ liệu biển số xe chuẩn Việt Nam (biển trắng, biển vàng kinh doanh, biển xanh công vụ, biển đỏ quân đội) với đầy đủ các hiệu ứng camera hành trình (rung nhòe chuyển động, mờ do lấy nét, nhiễu cảm biến, nghiêng phối cảnh 3D):
```bash
python generate_synth_plates.py --num_train 9000 --num_val 1000 --output_dir data_source/dataset_ocr
```

#### 2.2. Cắt biển số thực tế từ ảnh xe với Smart Padding
Sử dụng mô hình YOLO11n đã huấn luyện ở Bước 1 để tự động cắt các biển số thực tế, bổ sung tỷ lệ biên lề an toàn 8% để hỗ trợ STN:
```bash
python crop_plates.py --weights runs/train/yolo11n_license_plate/weights/best.pt --output_dir data_source/real_cropped_plates
```

---

### Bước 3: Huấn luyện Mạng STN + LPRNet OCR

Mô hình được huấn luyện end-to-end với CTC Loss kết hợp cơ chế nạp trước vào RAM (**In-Memory Caching** ~70 MB) giúp tăng tốc độ huấn luyện gấp 15 lần:

```bash
# Huấn luyện với kiến trúc Bounded STN (Khuyến nghị - Cho kết quả tối ưu nhất)
python train_lprnet.py --bounded_stn --epochs 30 --batch_size 64 --lr 0.001 --data_dir data_source/dataset_ocr
```

Các chế độ huấn luyện hỗ trợ:
- `--bounded_stn`: Bật tính năng giới hạn góc Affine bằng hàm `torch.tanh`, chống hiện tượng sụp đổ gradient (CTC Loss Collapse).
- `--two_stage`: Huấn luyện 2 giai đoạn (Giai đoạn 1 khóa trọng số STN, Giai đoạn 2 mở khóa toàn bộ mạng).

---

### Bước 4: So sánh & Đánh giá Thực nghiệm (Experiment Evaluation)

Mở notebook [notebook/compare_lprnet_experiments.ipynb](file:///d:/GitHub/Tiny-model-for-number-plate-recognization/notebook/compare_lprnet_experiments.ipynb) để trích xuất số liệu báo cáo:
- Đồ thị so sánh đường cong Loss (Train Loss & Val Loss) giữa các phương án.
- Bảng tổng hợp các chỉ số quan trọng: Best Epoch, Validation Loss, Character Accuracy, Plate Accuracy.
- Trực quan hóa ảnh đầu vào, ảnh sau khi STN tự động nắn thẳng và kết quả giải mã CTC thực tế.

---

### Bước 5: Ghép nối Pipeline Hoàn chỉnh (End-to-End ALPR)

Chạy inference tích hợp toàn diện trên video hoặc chuỗi frame trích xuất từ camera hành trình:

```bash
# Chạy trên một file video dashcam cụ thể:
python pipeline_end2end.py --video "data_source/sample_dashcam.mp4" --output_dir output

# Hoặc chạy trên thư mục chứa chuỗi frame ảnh:
python pipeline_end2end.py --frames_dir "data_source/organized_frames_dataset/1f652a87_447_clip_005_0027_0034_N" --output_dir output
```

**Các tham số tùy chọn:**
- `--vehicle_weights`: Đường dẫn mô hình xe (mặc định: `yolo11n.pt`).
- `--plate_weights`: Đường dẫn mô hình biển số (mặc định: `runs/train/yolo11n_license_plate/weights/best.pt`).
- `--lprnet_weights`: Đường dẫn mô hình OCR LPRNet (mặc định: `runs/train/lprnet_bounded_stn/weights/best.pt`).
- `--conf_vehicle`: Ngưỡng tin cậy phát hiện xe (mặc định: `0.35`).
- `--conf_plate`: Ngưỡng tin cậy phát hiện biển số (mặc định: `0.30`).
- `--window_size`: Cửa sổ khung hình bỏ phiếu đa số theo thời gian (mặc định: `15`).

---

## 🔬 Đột phá Kỹ thuật & Kinh nghiệm Triển khai

| Vấn đề Thực tế (Thử thách) | Nguyên nhân cốt lõi | Giải pháp Kỹ thuật Đã Thực hiện |
| :--- | :--- | :--- |
| **Sụp đổ CTC Loss (Plate Acc = 0%)** | Lớp `container` cuối cùng chứa `ReLU + BatchNorm` chặn logits âm, triệt tiêu gradient CTC; STN không giới hạn phóng to tọa độ tới vô cực ($128\times$). | Loại bỏ `ReLU/BatchNorm` tại projection layer; bổ sung `torch.tanh` ràng buộc ma trận Affine xoay tối đa $\pm 14^\circ$, co giãn $0.8 - 1.2\times$. |
| **Watermark / Timestamp nhận nhầm biển số** | Góc dưới camera hành trình 2K có chữ số ngày giờ màu vàng/trắng tương tự biển số xe. | **Cascaded 2-Stage**: Chỉ tìm kiếm biển số bên trong bounding box xe đã được phát hiện và theo dõi bởi ByteTrack. |
| **Tốc độ đọc đĩa chậm khi huấn luyện** | Hàng chục nghìn ảnh nhỏ liên tục nạp từ ổ cứng cơ/SSD qua từng epoch. | **In-Memory RAM Preloading**: Nạp toàn bộ 10,000 ảnh tiền xử lý vào RAM (~70 MB), rút ngắn mỗi epoch từ 5 phút xuống còn ~20 giây trên CPU. |
| **Rung nhòe khung hình camera hành trình** | Xe di chuyển qua gờ giảm tốc hoặc ban đêm gây mờ cục bộ một vài frame. | **Temporal Majority Voting**: Thuật toán gom nhóm ký tự theo `track_id` và bầu chọn theo tần suất, độ chính xác tăng lũy tiến theo độ dài video. |

---

## 📊 Bảng Thông số Kỹ thuật Mô hình (Model Specs)

| Mô hình | Đầu vào (Input Size) | Số lượng Tham số (Params) | Dung lượng Tệp (Weight Size) | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (COCO)** | $640 \times 640 \times 3$ | $2.6 \text{ M}$ | $\sim 5.2 \text{ MB}$ | Phát hiện và phân loại xe (Car, Bus, Truck, Motor) |
| **YOLO11n-Plate** | $640 \times 640 \times 3$ | $2.6 \text{ M}$ | $\sim 5.2 \text{ MB}$ | Định vị bounding box biển số xe bên trong vùng xe |
| **STN + LPRNet** | $94 \times 24 \times 3$ | $1.08 \text{ M}$ | $\sim 4.15 \text{ MB}$ | Nắn chỉnh hình học & Nhận diện chuỗi ký tự biển số |
| **Tổng thể Pipeline** | $1920 \times 1080 / 2\text{K}$ | $\sim 6.3 \text{ M}$ | **$< 15 \text{ MB}$** | Hệ thống ALPR hoàn chỉnh thời gian thực |
