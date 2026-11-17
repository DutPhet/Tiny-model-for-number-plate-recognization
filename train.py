import os
from pathlib import Path
from ultralytics import YOLO

def train_license_plate_model(
    data_yaml="data.yaml",
    model_weights="yolo11n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    project_name="runs/train",
    experiment_name="yolo11n_license_plate"
):
    """
    Huấn luyện mô hình YOLO11n trên dataset biển số xe (được định nghĩa trong data.yaml).
    """
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / data_yaml
    weights_path = base_dir / model_weights

    if not data_path.exists():
        print(f"❌ Không tìm thấy file cấu hình data: {data_path}")
        return

    print("=" * 60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN YOLO11n - LICENSE PLATE DETECTION")
    print(f"📄 Data config: {data_path}")
    print(f"🎯 Model base:   {weights_path}")
    print(f"⚙️ Epochs:      {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print("=" * 60)

    # 1. Tải mô hình nền YOLO11n
    model = YOLO(str(weights_path))

    # 2. Khởi chạy quá trình huấn luyện
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(base_dir / project_name),
        name=experiment_name,
        exist_ok=True,
        plots=True,
        verbose=True
    )

    best_model_path = base_dir / project_name / experiment_name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"✅ Huấn luyện hoàn tất!")
    if best_model_path.exists():
        print(f"🏆 Trọng số tốt nhất đã lưu tại: {best_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_license_plate_model(
        data_yaml="data.yaml",
        model_weights="yolo11n.pt",
        epochs=30,      # Bạn có thể tăng lên 50 - 100 nếu muốn mô hình đạt độ chính xác cao hơn
        imgsz=640,
        batch=8         # Điều chỉnh tùy theo dung lượng VRAM/RAM của máy
    )
