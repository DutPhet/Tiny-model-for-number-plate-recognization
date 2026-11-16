import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Các lớp phương tiện trong bộ dữ liệu COCO:
# 2: car (ô tô), 3: motorcycle (xe máy), 5: bus (xe buýt), 7: truck (xe tải)
VEHICLE_CLASSES = [2, 3, 5, 7]

def auto_label_and_track_dual(
    dataset_dir,
    output_dir="output",
    vehicle_model_path="yolo11n.pt",
    plate_model_path="runs/train/yolo11n_license_plate/weights/best.pt",
    tracker="bytetrack.yaml",
    conf_vehicle=0.25,
    conf_plate=0.20,
    include_track_id=True,
    copy_images=True
):
    """
    Duyệt qua các folder clip trong dataset, chạy tracking đồng thời:
    - Track XE (vehicle_model)
    - Track BIỂN SỐ (plate_model)
    TẤT CẢ KẾT QUẢ ĐƯỢC XUẤT VÀO THƯ MỤC OUTPUT:
    - Ảnh gốc được copy vào output/<clip_name>/
    - File nhãn xe: output/<clip_name>/<frame>_vehicle.txt
    - File nhãn biển: output/<clip_name>/<frame>_plate.txt
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"❌ Thư mục dataset không tồn tại: {dataset_dir.resolve()}")
        return

    # 1. Tải mô hình Xe và Biển số
    print(f"🔄 Đang tải mô hình XE:      {vehicle_model_path}")
    vehicle_model = YOLO(vehicle_model_path)
    
    print(f"🔄 Đang tải mô hình BIỂN SỐ: {plate_model_path}")
    plate_model = YOLO(plate_model_path)
    
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n🚀 Đang xử lý tập: {split.upper()} -> Xuất vào: {output_dir.resolve()}")
        clip_folders = [f for f in split_dir.iterdir() if f.is_dir()]

        for clip_folder in tqdm(clip_folders, desc=f"Processing {split}"):
            # Lấy tất cả file ảnh trong folder clip
            image_extensions = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(str(clip_folder / ext)))

            if not image_paths:
                continue

            # Sắp xếp ảnh theo thứ tự thời gian
            image_paths = sorted(image_paths, key=lambda x: Path(x).name)

            # Tạo folder cho clip trong thư mục output
            out_clip_dir = output_dir / clip_folder.name
            out_clip_dir.mkdir(parents=True, exist_ok=True)

            # Copy ảnh gốc vào thư mục output nếu được cấu hình
            if copy_images:
                for img_p in image_paths:
                    dest_p = out_clip_dir / Path(img_p).name
                    if not dest_p.exists():
                        shutil.copy2(img_p, dest_p)

            # -------------------------------------------------------------
            # BƯỚC 1: Tracking XE trên chuỗi frame
            # -------------------------------------------------------------
            vehicle_results = vehicle_model.track(
                source=image_paths,
                persist=True,
                tracker=tracker,
                classes=VEHICLE_CLASSES,
                conf=conf_vehicle,
                stream=True,
                verbose=False
            )

            for result in vehicle_results:
                img_name = Path(result.path).stem
                vehicle_txt_path = out_clip_dir / f"{img_name}_vehicle.txt"
                
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    with open(vehicle_txt_path, 'w', encoding='utf-8') as f:
                        pass
                else:
                    xywhn = boxes.xywhn.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    track_ids = (
                        boxes.id.cpu().numpy().astype(int) 
                        if boxes.id is not None 
                        else [-1] * len(boxes)
                    )
                    lines = []
                    for (x, y, w, h), cls_id, track_id in zip(xywhn, cls_ids, track_ids):
                        if include_track_id:
                            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {track_id}\n")
                        else:
                            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    with open(vehicle_txt_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

            # -------------------------------------------------------------
            # BƯỚC 2: Tracking BIỂN SỐ trên chuỗi frame
            # -------------------------------------------------------------
            plate_results = plate_model.track(
                source=image_paths,
                persist=True,
                tracker=tracker,
                conf=conf_plate,
                stream=True,
                verbose=False
            )

            for result in plate_results:
                img_name = Path(result.path).stem
                plate_txt_path = out_clip_dir / f"{img_name}_plate.txt"
                
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    with open(plate_txt_path, 'w', encoding='utf-8') as f:
                        pass
                else:
                    xywhn = boxes.xywhn.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    track_ids = (
                        boxes.id.cpu().numpy().astype(int) 
                        if boxes.id is not None 
                        else [-1] * len(boxes)
                    )
                    lines = []
                    for (x, y, w, h), cls_id, track_id in zip(xywhn, cls_ids, track_ids):
                        if include_track_id:
                            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {track_id}\n")
                        else:
                            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    with open(plate_txt_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

    print(f"\n✅ Đã lưu toàn bộ kết quả vào thư mục OUTPUT: {output_dir.resolve()}")
    print("   Trong mỗi folder clip chứa:")
    print("   - Ảnh gốc .jpg")
    print("   - File nhãn xe:       <tên_ảnh>_vehicle.txt")
    print("   - File nhãn biển số:  <tên_ảnh>_plate.txt")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # Đường dẫn dataset nguồn
    possible_paths = [
        BASE_DIR / "data_source" / "organized_frames_dataset",
        BASE_DIR / "organized_frames_dataset",
        Path("../data_source/organized_frames_dataset")
    ]
    DATASET_DIR = next((p for p in possible_paths if p.exists()), possible_paths[0])
    
    # Thư mục OUTPUT đầu ra
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Model XE (yolo11n.pt)
    VEHICLE_WEIGHTS = str(BASE_DIR / "yolo11n.pt")
    
    # Model BIỂN SỐ (best.pt vừa train xong)
    PLATE_WEIGHTS = str(BASE_DIR / "runs" / "train" / "yolo11n_license_plate" / "weights" / "best.pt")
    if not Path(PLATE_WEIGHTS).exists():
        PLATE_WEIGHTS = VEHICLE_WEIGHTS

    TRACKER_TYPE = "bytetrack.yaml"
    CONF_VEHICLE = 0.25
    CONF_PLATE = 0.20

    print("=" * 65)
    print("🚗 & 🏷️  AUTO TRACK & LABEL -> LƯU KẾT QUẢ VÀO THƯ MỤC OUTPUT")
    print(f"📂 Dataset nguồn:   {DATASET_DIR}")
    print(f"📁 Thư mục OUTPUT:  {OUTPUT_DIR}")
    print(f"🚗 Model Xe:        {VEHICLE_WEIGHTS}")
    print(f"🏷️  Model Biển số:   {PLATE_WEIGHTS}")
    print("=" * 65)

    auto_label_and_track_dual(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        vehicle_model_path=VEHICLE_WEIGHTS,
        plate_model_path=PLATE_WEIGHTS,
        tracker=TRACKER_TYPE,
        conf_vehicle=CONF_VEHICLE,
        conf_plate=CONF_PLATE,
        include_track_id=True,
        copy_images=True
    )
