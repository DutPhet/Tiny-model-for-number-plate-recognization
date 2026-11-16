# Tiny Model for Number Plate Recognition & Tracking (Edge ALPR)

Dự án phát hiện và theo dõi biển số xe tự động ứng dụng kiến trúc **YOLO11n** kết hợp thuật toán tracking **ByteTrack**, tối ưu hóa cho camera hành trình và thiết bị biên (Edge Devices).

---

## 📁 Cấu trúc Dự án (Project Structure)

```text
Tiny-model-for-number-plate-recognization/
├── data_source/                                # Dữ liệu đầu vào
│   ├── Vietnam license-plate.v1i.yolov8/       # Dataset huấn luyện phát hiện biển số (train/valid/test)
│   ├── License Plate Detection.v1i.yolov8/     # Dataset dự phòng
│   └── organized_frames_dataset/               # Dữ liệu clip frame từ camera hành trình
├── notebook/                                   # Thư mục notebook báo cáo
│   └── view_boxes.ipynb                        # Notebook trực quan hóa và kiểm tra kết quả
├── output/                                     # Kết quả gán nhãn & tracking (ảnh và txt theo từng folder clip)
├── runs/                                       # Kết quả huấn luyện YOLO
│   └── train/yolo11n_license_plate/weights/    # Trọng số tốt nhất (best.pt) sau khi train
├── auto_track_label.py                         # Pipeline tự động tracking & gán nhãn đồng thời (Xe + Biển số)
├── data.yaml                                   # Cấu hình dataset chuẩn YOLO (đường dẫn tương đối)
├── requirements.txt                            # Danh sách thư viện phụ thuộc
├── train.py                                    # Script huấn luyện YOLO11n phát hiện biển số xe
├── view_boxes.ipynb                            # Notebook trực quan hóa tại thư mục gốc
└── yolo11n.pt                                  # Model COCO cơ sở (phát hiện xe)
```

---

## ⚙️ Cài đặt Môi trường (Installation)

Yêu cầu Python 3.10+ (khuyến nghị sử dụng virtual environment):

```bash
# Tạo môi trường ảo
python -m venv myenv

# Kích hoạt môi trường (Windows PowerShell)
.\myenv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

> **Lưu ý**: Dự án đã tích hợp thư viện `lapx` để hỗ trợ giải thuật Hungarian/Linear Assignment cho bộ theo dõi **ByteTrack**.

---

## 🚀 Quy trình Thực thi (Pipeline Workflow)

### Bước 1: Huấn luyện mô hình phát hiện biển số xe (YOLO11n)
Mô hình nền `yolo11n.pt` được huấn luyện thêm (fine-tune) trên tập dữ liệu biển số xe Việt Nam:

```bash
python train.py
```
- **Cấu hình data**: `data.yaml` tự động trỏ đến `data_source/Vietnam license-plate.v1i.yolov8`.
- **Trọng số đầu ra**: Được lưu tự động tại `runs/train/yolo11n_license_plate/weights/best.pt`.

---

### Bước 2: Tự động Gán nhãn & Tracking Chuỗi Frame (Auto Label & Tracking)
Chạy tracking song song hai mô hình (Xe + Biển số) trên dữ liệu camera hành trình `data_source/organized_frames_dataset`:

```bash
python auto_track_label.py
```
- **Mô hình Xe**: `yolo11n.pt` (lọc các class: Car, Motorcycle, Bus, Truck).
- **Mô hình Biển số**: `runs/train/yolo11n_license_plate/weights/best.pt`.
- **Tracker**: `bytetrack.yaml` (gán và duy trì `track_id` ổn định qua các khung hình).
- **Đóng gói đầu ra**:
  - Lưu vào thư mục `output/<tên_clip>/`.
  - Mỗi khung hình được đóng cặp đồng bộ:
    - Ảnh gốc: `<tên_ảnh>.jpg`
    - Nhãn xe: `<tên_ảnh>_vehicle.txt` (`<class_id> <x_center> <y_center> <width> <height> <track_id>`)
    - Nhãn biển số: `<tên_ảnh>_plate.txt` (`<class_id> <x_center> <y_center> <width> <height> <track_id>`)

---

### Bước 3: Kiểm tra & Trực quan hóa Kết quả (Visualization & Report)
Mở notebook `view_boxes.ipynb` (hoặc `notebook/view_boxes.ipynb`) bằng Jupyter Notebook / VS Code:

- Tự động phát hiện đường dẫn gốc dự án không phụ thuộc ổ đĩa (`PROJECT_ROOT`).
- Đọc và vẽ bounding box trực tiếp từ thư mục `output/`:
  - 🚗 **Xe**: Box màu xanh lá kèm Track ID và tên loại xe.
  - 🏷️ **Biển số**: Box màu đỏ cam kèm Track ID.
- Thống kê chi tiết kích thước pixel và tọa độ bounding box cho từng khung hình để báo cáo.
