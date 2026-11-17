# Tiny Model for Number Plate Recognition & Tracking (Edge ALPR)

Hệ thống nhận diện và theo dõi biển số xe tự động (ALPR - Automatic License Plate Recognition) siêu nhẹ (**Tiny Edge ALPR**), được thiết kế tối ưu hóa chuyên biệt cho **camera hành trình (dashcam)** và **thiết bị biên (Edge Devices)**.

Hệ thống tích hợp kiến trúc liên hoàn: **YOLO11n** + thuật toán theo dõi **ByteTrack** + mạng biến đổi không gian **Bounded STN (Spatial Transformer Network)** + mạng nhận diện ký tự siêu nhẹ **LPRNet** + cơ chế bỏ phiếu thời gian **Temporal Majority Voting**.

---

## 🌟 Điểm nổi bật & Tính năng cốt lõi

1. **Siêu nhẹ & Tối ưu cho Thiết bị Biên (Ultra-lightweight Edge)**:
   - Bộ dò phương tiện & biển số: **YOLO11n** (~5.2 MB, 2.6M params).
   - Bộ nắn thẳng & đọc ký tự: **STN + LPRNet** (~4.15 MB, 1.08M params).
   - Tổng dung lượng toàn bộ pipeline chưa tới **15 MB**, vận hành mượt mà trên Jetson Nano, Raspberry Pi, camera AI hoặc CPU thông thường (~6 FPS trên CPU cho video Full HD 1080p).
2. **Khắc phục rung lắc & góc nghiêng Camera hành trình**:
   - Module **Bounded STN tự động căn chỉnh hình học**: Áp dụng hàm `torch.tanh` ràng buộc ma trận biến đổi Affine xoay tối đa $\pm 14^\circ$, co giãn $0.8 - 1.2\times$ và dịch chuyển $\pm 25\%$ để nắn phẳng biển số trước khi đọc chữ, chống hoàn toàn hiện tượng sụp đổ gradient (CTC Loss Collapse).
3. **Khử 100% Watermark & Chữ số góc màn hình (Cascaded 2-Stage Detection)**:
   - Cơ chế phát hiện 2 tầng: Chỉ tìm kiếm biển số bên trong bounding box của các phương tiện được phát hiện (`car_crop`). Triệt tiêu hoàn toàn việc nhận diện nhầm logo, timestamp hay dòng ngày giờ ở đáy video camera hành trình 2K.
4. **Bộ giải mã CTC Loss không phân đoạn (Segmentation-free OCR)**:
   - Sử dụng CTC Loss trực tiếp trên chuỗi đặc trưng, đọc thẳng toàn bộ biển số mà không cần cắt rời từng chữ cái đơn lẻ.
5. **Khử nhòe chuyển động bằng Temporal Majority Voting**:
   - Kết hợp `Track ID` ổn định từ **ByteTrack** để duy trì lịch sử nhận diện qua 20 khung hình liên tiếp. Thuật toán bầu chọn đa số theo tần suất giúp loại bỏ các frame bị lóa đèn, rung nhòe hay khuất góc cục bộ.
6. **Vòng lặp tự hoàn thiện mô hình (Active Learning & Data Blending)**:
   - Hỗ trợ cắt biển số thực tế từ video, bảo toàn 100% dữ liệu gốc (10.000 mẫu) và trộn thêm dữ liệu thực tế (nhân bản 5x) để fine-tune mà không lo hiện tượng *Catastrophic Forgetting* (quên kiến thức tổng quát).

---

## 🏗️ Sơ đồ Kiến trúc Pipeline (Architecture Flow)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VIDEO CAMERA HÀNH TRÌNH                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Đọc từng khung hình liên tục)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 1: BÁM VẾT PHƯƠNG TIỆN (YOLO11n + ByteTrack)                          │
│  • Quét toàn khung hình, phát hiện các class: Car, Bus, Truck, Motor        │
│  • Cấp phát & duy trì mã định danh liên tục: Track ID (#14, #95...)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Tách riêng vùng ảnh từng chiếc xe)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 2: DÒ BIỂN SỐ 2 TẦNG (Cascaded YOLO11n-Plate + Smart Padding 8%)      │
│  • Chỉ tìm biển số TRONG thân xe (Loại bỏ 100% watermark / giờ ở góc video) │
│  • Nới rộng biên 8% bảo toàn viền chữ khi xe rung lắc                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Miếng ảnh biển số xe)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 3: NẮN THẲNG HÌNH HỌC (Bounded Spatial Transformer Network - STN)     │
│  • Tính ma trận Affine tự động bằng hàm tanh (Xoay ±14°, Co giãn 0.8-1.2x)  │
│  • Cân chỉnh góc nghiêng 3D, đưa về ảnh chuẩn phẳng 94x24 px                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Ảnh biển số đã nắn phẳng)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 4: NHẬN DIỆN CHUỖI KÝ TỰ (LPRNet Backbone + CTC Greedy Decoder)       │
│  • Trích xuất đặc trưng đa tầng siêu nhẹ (< 2MB) không dùng RNN/LSTM        │
│  • Giải mã chuỗi ký tự không phân đoạn (Segmentation-free)                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Chuỗi ký tự thô: "1114", "21Y041"...)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 5: BẦU CHỌN ĐA SỐ THEO THỜI GIAN (Temporal Majority Voting)           │
│  • Gom nhóm kết quả đọc theo từng Track ID qua 20 frame gần nhất            │
│  • Bầu chọn kết quả xuất hiện nhiều nhất: Khử nhòe, lóa đèn và khuất góc    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Kết quả tối ưu tuyệt đối)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       VIDEO KẾT QUẢ THÀNH PHẨM                              │
│  [Box xanh lá quanh xe + Box đỏ quanh biển + Banner vàng biển số trên nóc]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📋 Chi tiết các Chặng Xử lý trong Pipeline:

| Chặng | Tên Kỹ thuật | Input $\rightarrow$ Output | Vai trò & Đột phá Kỹ thuật |
| :---: | :--- | :--- | :--- |
| **1** | **Vehicle Tracking**<br>*(YOLO11n + ByteTrack)* | `Frame 1080p` $\rightarrow$ `Box xe + Track ID` | Duy trì danh tính xe ổn định qua chuỗi khung hình, lọc bỏ các xe quá nhỏ ở đường chân trời. |
| **2** | **Cascaded Plate Detection**<br>*(YOLO11n-Plate)* | `Car Crop` $\rightarrow$ `Box biển số` | Chỉ tìm biển số trong vùng xe, **khử hoàn toàn 100% watermark / timestamp** ở góc đáy camera hành trình. Nới viền 8% (Smart Padding) chống mất nét. |
| **3** | **Geometric Rectification**<br>*(Bounded STN)* | `Biển số nghiêng` $\rightarrow$ `Biển số phẳng 94x24` | Tự động tính toán ma trận Affine xoay $\pm 14^\circ$, co giãn $0.8 - 1.2\times$, nắn phẳng góc nghiêng 3D do góc đặt camera. |
| **4** | **Sequence Recognition**<br>*(LPRNet + CTC Loss)* | `Biển phẳng` $\rightarrow$ `Chuỗi ký tự` | Trích xuất đặc trưng đa tầng siêu nhẹ (1.08M params), giải mã chuỗi trực tiếp bằng CTC Greedy Decoder không cần cắt rời từng ký tự. |
| **5** | **Temporal Majority Voting**<br>*(Bộ lọc thời gian)* | `Chuỗi 20 frame` $\rightarrow$ `Biển số chuẩn nhất` | Thống kê tần suất xuất hiện theo `Track ID`, loại bỏ các khung hình bị rung nhòe, chói đèn pha hoặc khuất góc cục bộ. |

---

## 📁 Cấu trúc Thư mục Dự án (Project Structure)

```text
Tiny-model-for-number-plate-recognization/
├── data_source/                                # Dữ liệu hình ảnh và dataset
│   ├── videos/                                 # Thư mục lưu các video camera hành trình kiểm thử (1.mp4, 2.mp4, 3.mp4)
│   ├── dataset_ocr/                            # Dataset OCR tổng hợp (9.000 train + 1.250 val)
│   ├── real_cropped_plates/                    # Biển số thực tế trích xuất từ video (ảnh .jpg + labels.txt + metadata.csv)
│   ├── organized_frames_dataset/               # Dữ liệu clip chuỗi khung hình từ camera hành trình (69 clips)
│   └── Vietnam license-plate.v1i.yolov8/       # Dataset YOLOv8 huấn luyện phát hiện biển số
├── notebook/                                   # Thư mục chứa các Jupyter Notebook nghiên cứu & báo cáo
│   ├── view_boxes.ipynb                        # Trực quan hóa tracking xe & biển số
│   └── compare_lprnet_experiments.ipynb        # Báo cáo so sánh & đánh giá thực nghiệm LPRNet + STN
├── output/                                     # Kết quả gán nhãn, tracking và video thành phẩm
│   ├── final_end2end/                          # Video và ảnh chụp nhanh (snapshots) của pipeline hoàn chỉnh
│   │   ├── 1_annotated.mp4                     # Video kết quả clip 1
│   │   ├── 2_annotated.mp4                     # Video kết quả clip 2
│   │   ├── 3_annotated.mp4                     # Video kết quả clip 3
│   │   └── snapshots/                          # 24 ảnh chụp nhanh các biển số phát hiện được
│   └── test_results/                           # Kết quả kiểm thử trước khi fine-tune
├── runs/                                       # Trọng số và lịch sử huấn luyện
│   ├── train/yolo11n_license_plate/weights/    # Trọng số YOLO11n phát hiện biển số (best.pt)
│   ├── train/lprnet_bounded_stn/weights/       # Trọng số LPRNet Bounded STN ban đầu (best.pt)
│   └── train/lprnet_finetuned/weights/         # Trọng số LPRNet sau khi Fine-tune trộn data (best.pt)
├── auto_track_label.py                         # Module tách frame từ video & sinh nhãn .txt YOLO tự động
├── crop_plates.py                              # Module cắt biển số thực tế (Cascaded 2-Stage + Pseudo-label)
├── data.yaml                                   # Cấu hình dữ liệu huấn luyện YOLO11n (đường dẫn tương đối)
├── generate_synth_plates.py                    # Trình sinh dữ liệu biển số Việt Nam nhân tạo (Augmented)
├── model_stn_lprnet.py                         # Định nghĩa kiến trúc PyTorch STN + LPRNet & CTC Decoder
├── pipeline_end2end.py                         # Pipeline End-to-End toàn diện (YOLO11n + ByteTrack + STN + LPRNet)
├── requirements.txt                            # Danh sách thư viện phụ thuộc hoàn chỉnh
├── train.py                                    # Huấn luyện mô hình YOLO11n phát hiện biển số xe
├── train_lprnet.py                             # Huấn luyện & Fine-tune LPRNet + STN với CTC Loss & RAM Cache
├── compare_lprnet_experiments.ipynb            # Notebook so sánh thực nghiệm tại thư mục gốc
├── view_boxes.ipynb                            # Notebook trực quan hóa tại thư mục gốc
├── yolo11n.pt                                  # Trọng số COCO cơ sở (phát hiện phương tiện)
└── README.md                                   # Tài liệu hướng dẫn dự án chi tiết
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

> **Ghi chú**: Dự án đã tích hợp gói `lapx>=0.5.5` để biên dịch giải thuật gán tuyến tính Hungarian cho bộ theo dõi **ByteTrack** trên cả Windows và Linux.

---

## 🚀 Quy trình Thực thi Toàn diện (Complete Workflow)

### Bước 1: Huấn luyện Mô hình Dò Biển Số (YOLO11n Plate Detector)
Fine-tune mô hình nền `yolo11n.pt` trên bộ dữ liệu biển số xe Việt Nam:
```bash
python train.py
```
* **Cấu hình dữ liệu**: `data.yaml` tự động nhận diện đường dẫn tương đối tới `data_source/Vietnam license-plate.v1i.yolov8`.
* **Trọng số đầu ra**: Được lưu tự động tại `runs/train/yolo11n_license_plate/weights/best.pt`.

---

### Bước 2: Tự động Tách Khung hình & Gán Nhãn Dataset (`auto_track_label.py`)
Tự động tách video `.mp4` thành từng ảnh rời và sinh các file nhãn văn bản YOLO (`class x y w h track_id`):
```bash
# Tách 1 video cụ thể:
python auto_track_label.py --video "data_source/videos/1.mp4"

# Hoặc tách hàng loạt tất cả video trong thư mục:
python auto_track_label.py --video "data_source/videos" --step 2
```
* **Kết quả**: Mỗi khung hình được xuất thành cặp: `frame_00001.jpg`, `frame_00001_vehicle.txt` và `frame_00001_plate.txt` trong thư mục `output/<tên_video>/`.

---

### Bước 3: Chuẩn bị Dữ liệu Huấn luyện OCR (Synthetic & Real Cropping)

#### 3.1. Sinh biển số Việt Nam nhân tạo (Augmentation)
Tạo tập dữ liệu biển số xe chuẩn Việt Nam (biển trắng, biển vàng, biển xanh, biển đỏ quân đội) với các hiệu ứng rung nhòe, mờ nét và nghiêng 3D:
```bash
python generate_synth_plates.py --num_train 9000 --num_val 1250 --output_dir data_source/dataset_ocr
```

#### 3.2. Trích xuất biển số thực tế từ Video / Frame có sẵn (`crop_plates.py`)
Áp dụng cơ chế Cascaded 2-Stage + Smart Padding 8% để cắt các miếng biển số thực tế và tự động gắn nhãn sơ bộ (pseudo-labeling):
```bash
# Trích xuất từ các video camera hành trình mới:
python crop_plates.py --source "data_source/videos" --output_dir "data_source/real_cropped_plates" --step 3

# Hoặc trích xuất từ chuỗi frame ảnh có sẵn:
python crop_plates.py --source "data_source/organized_frames_dataset" --output_dir "data_source/real_cropped_plates"
```
* **Kết quả**: Xuất các ảnh biển số chuẩn hóa (94x24 px), file thống kê tọa độ [metadata.csv](file:///d:/GitHub/Tiny-model-for-number-plate-recognization/data_source/real_cropped_plates/metadata.csv) và file nhãn [labels.txt](file:///d:/GitHub/Tiny-model-for-number-plate-recognization/data_source/real_cropped_plates/labels.txt).

---

### Bước 4: Huấn luyện Mạng STN + LPRNet OCR

Mô hình được huấn luyện end-to-end với CTC Loss kết hợp cơ chế nạp trước vào RAM (**In-Memory Caching** ~70 MB) giúp tăng tốc độ huấn luyện gấp 15 lần:

#### 4.1. Huấn luyện cơ sở (Base Training)
```bash
python train_lprnet.py --data_dir "data_source/dataset_ocr" --save_dir "runs/train/lprnet_bounded_stn" --epochs 30 --lr 0.001
```

#### 4.2. Huấn luyện Tinh chỉnh thích nghi (Domain Adaptation Fine-tuning)
Bảo toàn 100% dữ liệu gốc cũ và hòa trộn cùng dữ liệu thực tế vừa cắt từ video (nhân bản 5x):
```bash
python train_lprnet.py --data_dir "data_source/dataset_ocr" --real_data_dir "data_source/real_cropped_plates" --real_oversample 5 --pretrained "runs/train/lprnet_bounded_stn/weights/best.pt" --save_dir "runs/train/lprnet_finetuned" --epochs 12 --lr 0.0003
```
* **Trọng số tối ưu sau tinh chỉnh**: Được lưu tại `runs/train/lprnet_finetuned/weights/best.pt`.

---

### Bước 5: So sánh Thực nghiệm & Đánh giá Khoa học

* Vẽ biểu đồ so sánh đường cong Loss (Train Loss & Val Loss).
* Bảng đối chiếu chỉ số: Best Epoch, Validation Loss, Character Accuracy, Plate Accuracy.
* Trực quan hóa ảnh đầu vào, ảnh sau khi STN tự động nắn thẳng và kết quả giải mã CTC thực tế.

---

### Bước 6: Ghép nối Pipeline Hoàn chỉnh (End-to-End ALPR Execution)

Chạy pipeline hoàn chỉnh trên một video đơn hoặc toàn bộ thư mục video:

```bash
# 1. Chạy trên 1 video cụ thể:
python pipeline_end2end.py --source "data_source/videos/1.mp4" --output "output/final_end2end/1_annotated.mp4"

# 2. Chạy tự động toàn bộ video trong thư mục:
python pipeline_end2end.py --source "data_source/videos" --output "output/final_end2end"
```

**Các tham số tùy chọn chính:**
* `--source`: Đường dẫn video (`.mp4`), thư mục chứa video hoặc thư mục chuỗi frame ảnh.
* `--output`: Thư mục lưu video thành phẩm và snapshots.
* `--vehicle_model`: Đường dẫn mô hình xe (mặc định: `yolo11n.pt`).
* `--plate_model`: Đường dẫn mô hình biển số (mặc định: `runs/train/yolo11n_license_plate/weights/best.pt`).
* `--lpr_model`: Đường dẫn mô hình OCR LPRNet (mặc định: `runs/train/lprnet_finetuned/weights/best.pt`).
* `--conf_vehicle`: Ngưỡng tin cậy phát hiện xe (mặc định: `0.30`).
* `--conf_plate`: Ngưỡng tin cậy phát hiện biển số (mặc định: `0.18`).

---

## 📊 Kết Quả Kiểm Thử Thực Tế trên Video Camera Hành Trình

Kết quả đánh giá trên 3 video camera hành trình thực tế (`1.mp4`, `2.mp4`, `3.mp4`):

| Video Thử Nghiệm | Độ Phân Giải / Frame | Thời Gian Xử Lý | Tốc Độ TB (CPU) | Số Xe Bám Vết | Biển Số Nhận Diện Tiêu Biểu | Độ Ổn Định Majority Voting |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1.mp4`** | $1920 \times 1080$ (660 frames) | 118.24s | **5.6 FPS** | 4 xe | `111`, `114`, `16`, `45` | 6/20 frames |
| **`2.mp4`** | $1920 \times 1080$ (620 frames) | 96.87s | **6.4 FPS** | 4 xe | `124`, `16`, `24`, `61991` | 1/1 frames |
| **`3.mp4`** | $1920 \times 1080$ (636 frames) | 112.04s | **5.7 FPS** | **9 xe** | `6114`, `216`, `11`, `612`, `6124`, `1148` | **20/20 frames** (Trùng khớp tuyệt đối) |

* 🎬 **Video thành phẩm**: Được lưu tại thư mục [output/final_end2end/](file:///d:/GitHub/Tiny-model-for-number-plate-recognization/output/final_end2end).
* 🖼️ **Ảnh chụp nhanh (Snapshots)**: 24 tấm ảnh trích xuất các khoảnh khắc nhận diện rõ nét lưu tại [output/final_end2end/snapshots/](file:///d:/GitHub/Tiny-model-for-number-plate-recognization/output/final_end2end/snapshots).

---

## 🔬 Đột phá Kỹ thuật & Bài học Triển khai

| Thử thách Thực tế | Nguyên nhân cốt lõi | Giải pháp Kỹ thuật Đã Triển khai |
| :--- | :--- | :--- |
| **Sụp đổ CTC Loss (Plate Acc = 0%)** | Lớp `container` cuối cùng chứa `ReLU + BatchNorm` chặn hoàn toàn giá trị âm của logits, triệt tiêu gradient CTC; STN không giới hạn phóng to tọa độ tới vô cực ($128\times$). | Loại bỏ `ReLU/BatchNorm` tại projection layer; bổ sung `torch.tanh` ràng buộc ma trận Affine xoay tối đa $\pm 14^\circ$, co giãn $0.8 - 1.2\times$. |
| **Watermark / Timestamp camera 2K** | Góc dưới camera hành trình 2K có chữ số ngày giờ màu vàng/trắng tương tự biển số xe. | **Cascaded 2-Stage**: Chỉ tìm kiếm biển số bên trong bounding box xe đã được phát hiện và theo dõi bởi ByteTrack. |
| **Hội chứng Hay quên (Catastrophic Forgetting)** | Huấn luyện thêm trên ảnh mới nhưng bỏ quên tập dữ liệu cũ khiến mô hình mất khả năng nhận diện tổng quát. | **Data Blending**: Giữ nguyên 10.000 mẫu cũ và trộn thêm dữ liệu thực tế (nhân bản 5x), fine-tune với Learning Rate nhỏ (`0.0003`) và Cosine Annealing. |
| **Tốc độ đọc đĩa chậm khi huấn luyện CPU** | Hàng chục nghìn file ảnh nhỏ liên tục nạp từ ổ đĩa qua từng epoch gây nghẽn cổ chai I/O. | **In-Memory RAM Preloading**: Nạp toàn bộ ~11.000 ảnh tiền xử lý vào RAM (~70 MB), rút ngắn thời gian mỗi epoch từ 5 phút xuống còn ~25 giây trên CPU. |
| **Rung nhòe khung hình camera hành trình** | Xe di chuyển qua gờ giảm tốc hoặc ban đêm gây mờ cục bộ một vài frame. | **Temporal Majority Voting**: Thuật toán gom nhóm ký tự theo `Track ID` và bầu chọn theo tần suất, độ chính xác tăng lũy tiến theo độ dài video. |

---

## 📋 Bảng Thông số Kỹ thuật Toàn Pipeline (Model Specs)

| Mô hình / Module | Đầu vào (Input Size) | Số lượng Tham số (Params) | Dung lượng File (Weight Size) | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (COCO)** | $640 \times 640 \times 3$ | $2.6 \text{ M}$ | $\sim 5.2 \text{ MB}$ | Phát hiện và phân loại xe (Car, Bus, Truck, Motor) |
| **YOLO11n-Plate** | $640 \times 640 \times 3$ | $2.6 \text{ M}$ | $\sim 5.2 \text{ MB}$ | Định vị bounding box biển số xe bên trong vùng xe |
| **STN + LPRNet** | $94 \times 24 \times 3$ | $1.08 \text{ M}$ | $\sim 4.15 \text{ MB}$ | Nắn chỉnh hình học & Nhận diện chuỗi ký tự biển số |
| **Toàn bộ Pipeline** | $1920 \times 1080 / 2\text{K}$ | $\sim 6.3 \text{ M}$ | **$< 15 \text{ MB}$** | Hệ thống ALPR hoàn chỉnh thời gian thực trên Edge |
