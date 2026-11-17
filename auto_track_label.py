import os
import glob
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
from ultralytics import YOLO

# Các lớp phương tiện trong bộ dữ liệu COCO:
# 2: car (ô tô), 3: motorcycle (xe máy), 5: bus (xe buýt), 7: truck (xe tải)
VEHICLE_CLASSES = [2, 3, 5, 7]


def process_single_video(
    video_path,
    output_dir,
    vehicle_model,
    plate_model,
    tracker="bytetrack.yaml",
    conf_vehicle=0.25,
    conf_plate=0.20,
    include_track_id=True,
    frame_step=1
):
    """
    Cắt từng khung hình của 1 video thành ảnh .jpg rời,
    đồng thời chạy tracking Xe & Biển số để sinh nhãn .txt tương ứng.
    """
    video_p = Path(video_path)
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_p.resolve()}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_clip_dir = output_dir / video_p.stem
    out_clip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬 Đang cắt & gán nhãn video: {video_p.name} ({total_frames} frames, {w}x{h} px)")
    print(f"📁 Xuất ra thư mục: {out_clip_dir.resolve()}")

    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_p.name}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if frame_idx % frame_step != 0:
            continue

        frame_name = f"frame_{frame_idx:05d}"
        img_file = out_clip_dir / f"{frame_name}.jpg"
        veh_txt = out_clip_dir / f"{frame_name}_vehicle.txt"
        plate_txt = out_clip_dir / f"{frame_name}_plate.txt"

        # 1. Lưu ảnh gốc khung hình
        cv2.imwrite(str(img_file), frame)
        saved_count += 1

        # 2. Tracking XE
        veh_res = vehicle_model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            classes=VEHICLE_CLASSES,
            conf=conf_vehicle,
            verbose=False
        )

        lines_veh = []
        if veh_res and len(veh_res) > 0 and veh_res[0].boxes is not None and len(veh_res[0].boxes) > 0:
            boxes = veh_res[0].boxes
            xywhn = boxes.xywhn.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else [-1] * len(boxes)
            )
            for (bx, by, bw, bh), cls_id, tr_id in zip(xywhn, cls_ids, track_ids):
                if include_track_id:
                    lines_veh.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {tr_id}\n")
                else:
                    lines_veh.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

        with open(veh_txt, "w", encoding="utf-8") as f:
            f.writelines(lines_veh)

        # 3. Tracking BIỂN SỐ
        plate_res = plate_model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            conf=conf_plate,
            verbose=False
        )

        lines_plate = []
        if plate_res and len(plate_res) > 0 and plate_res[0].boxes is not None and len(plate_res[0].boxes) > 0:
            p_boxes = plate_res[0].boxes
            xywhn = p_boxes.xywhn.cpu().numpy()
            cls_ids = p_boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                p_boxes.id.cpu().numpy().astype(int)
                if p_boxes.id is not None
                else [-1] * len(p_boxes)
            )
            for (bx, by, bw, bh), cls_id, tr_id in zip(xywhn, cls_ids, track_ids):
                if include_track_id:
                    lines_plate.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {tr_id}\n")
                else:
                    lines_plate.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

        with open(plate_txt, "w", encoding="utf-8") as f:
            f.writelines(lines_plate)

    cap.release()
    pbar.close()
    print(f"✅ Hoàn tất video {video_p.name}: Đã xuất {saved_count} ảnh và file nhãn tương ứng.")


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
            image_extensions = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(str(clip_folder / ext)))

            if not image_paths:
                continue

            image_paths = sorted(image_paths, key=lambda x: Path(x).name)
            out_clip_dir = output_dir / clip_folder.name
            out_clip_dir.mkdir(parents=True, exist_ok=True)

            if copy_images:
                for img_p in image_paths:
                    dest_p = out_clip_dir / Path(img_p).name
                    if not dest_p.exists():
                        shutil.copy2(img_p, dest_p)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tự động cắt khung hình và gán nhãn Tracking (Xe + Biển số)")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn đến 1 video (.mp4) hoặc thư mục chứa các video")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Đường dẫn thư mục dataset đã có chuỗi frame (organized_frames_dataset)")
    parser.add_argument("--output_dir", type=str, default="output", help="Thư mục lưu kết quả xuất")
    parser.add_argument("--vehicle_weights", type=str, default="yolo11n.pt", help="Trọng số mô hình phát hiện xe")
    parser.add_argument("--plate_weights", type=str, default="runs/train/yolo11n_license_plate/weights/best.pt", help="Trọng số mô hình biển số")
    parser.add_argument("--conf_vehicle", type=float, default=0.25, help="Ngưỡng tự tin phát hiện xe")
    parser.add_argument("--conf_plate", type=float, default=0.20, help="Ngưỡng tự tin phát hiện biển số")
    parser.add_argument("--step", type=int, default=1, help="Bước nhảy cắt frame (1 = cắt mọi frame, 2 = cắt cách frame)")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)

    # 1. Nếu người dùng chỉ định video hoặc thư mục video
    if args.video:
        target_p = Path(args.video)
        veh_m = YOLO(args.vehicle_weights)
        plt_m = YOLO(args.plate_weights)

        if target_p.is_file() and target_p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            process_single_video(
                video_path=target_p,
                output_dir=out_dir,
                vehicle_model=veh_m,
                plate_model=plt_m,
                conf_vehicle=args.conf_vehicle,
                conf_plate=args.conf_plate,
                frame_step=args.step
            )
        elif target_p.is_dir():
            video_files = [p for p in target_p.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
            print(f"🎬 Tìm thấy {len(video_files)} video trong {target_p.resolve()}")
            for vid_f in video_files:
                process_single_video(
                    video_path=vid_f,
                    output_dir=out_dir,
                    vehicle_model=veh_m,
                    plate_model=plt_m,
                    conf_vehicle=args.conf_vehicle,
                    conf_plate=args.conf_plate,
                    frame_step=args.step
                )

    # 2. Nếu người dùng chạy trên dataset ảnh frame có sẵn
    else:
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else BASE_DIR / "data_source" / "organized_frames_dataset"
        auto_label_and_track_dual(
            dataset_dir=dataset_dir,
            output_dir=out_dir,
            vehicle_model_path=args.vehicle_weights,
            plate_model_path=args.plate_weights,
            conf_vehicle=args.conf_vehicle,
            conf_plate=args.conf_plate
        )
