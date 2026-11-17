"""
==============================================================================
Script: Real License Plate Cropper (Giai đoạn 1: Trích xuất Biển số Thật)
HỖ TRỢ TOÀN DIỆN CẢ 2 NGUỒN:
1. Video (.mp4, .avi, thư mục chứa video mới)
2. Chuỗi frame ảnh có sẵn (organized_frames_dataset, các thư mục clip con)
- Kiến trúc Cascaded 2-Stage: Dò xe (YOLO11n) -> Dò biển số trong thân xe (YOLO11n-Plate).
  -> Triệt tiêu hoàn toàn watermark / timestamp ở góc camera 2K.
- Smart Padding (nới biên lề an toàn 8%) tránh mất viền chữ khi xe rung lắc.
- Tự động chuẩn hóa về kích thước đầu vào của LPRNet (94x24 px).
- Sử dụng LPRNet đọc nhãn sơ bộ (pseudo-labeling) và xuất file metadata.csv + labels.txt.
==============================================================================
"""

import os
import glob
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO

from model_stn_lprnet import STN_LPRNet, CHARS, ctc_decode


VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck


def load_lpr_model(lpr_weights_path, device="cpu"):
    lpr_p = Path(lpr_weights_path)
    if not lpr_p.exists():
        return None
    class_num = len(CHARS) + 1
    model = STN_LPRNet(class_num=class_num, in_channels=3, use_stn=True).to(device)
    state = torch.load(str(lpr_p), map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def ocr_predict(lpr_model, plate_bgr, device="cpu"):
    if lpr_model is None or plate_bgr is None or plate_bgr.size == 0:
        return ""
    img_resized = cv2.resize(plate_bgr, (94, 24), interpolation=cv2.INTER_AREA)
    img_norm = (img_resized.astype(np.float32) - 127.5) * 0.0078125
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = lpr_model(img_tensor)
        preds = ctc_decode(logits, CHARS, blank_idx=0)
    return preds[0] if preds else ""


def crop_plates_from_frame(
    frame_bgr,
    veh_model,
    plate_model,
    lpr_model,
    conf_vehicle=0.30,
    conf_plate=0.18,
    padding_ratio=0.08,
    target_size=(94, 24),
    device="cpu"
):
    h_frame, w_frame = frame_bgr.shape[:2]
    tw, th = target_size
    crops_found = []

    # 1. Phát hiện xe
    veh_results = veh_model(frame_bgr, classes=VEHICLE_CLASSES, conf=conf_vehicle, verbose=False)
    if not veh_results or len(veh_results) == 0 or veh_results[0].boxes is None:
        return crops_found

    boxes = veh_results[0].boxes
    for box in boxes:
        v_xyxy = box.xyxy[0].cpu().numpy().astype(int)
        vx1, vy1, vx2, vy2 = v_xyxy
        vw, vh = vx2 - vx1, vy2 - vy1
        if vw < 50 or vh < 40:
            continue

        vx1_c = max(0, vx1)
        vy1_c = max(0, vy1)
        vx2_c = min(w_frame, vx2)
        vy2_c = min(h_frame, vy2)

        car_crop = frame_bgr[vy1_c:vy2_c, vx1_c:vx2_c]
        if car_crop.size == 0:
            continue

        # 2. Phát hiện biển số bên trong thân xe
        plate_results = plate_model(car_crop, conf=conf_plate, verbose=False)
        if not plate_results or len(plate_results) == 0 or plate_results[0].boxes is None:
            continue

        p_boxes = plate_results[0].boxes
        for p_box in p_boxes:
            conf_val = float(p_box.conf[0].item())
            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
            pw, ph = px2 - px1, py2 - py1
            if pw < 10 or ph < 6:
                continue

            # Smart Padding
            pad_w = int(pw * padding_ratio)
            pad_h = int(ph * padding_ratio)
            px1_pad = max(0, px1 - pad_w)
            py1_pad = max(0, py1 - pad_h)
            px2_pad = min(car_crop.shape[1], px2 + pad_w)
            py2_pad = min(car_crop.shape[0], py2 + pad_h)

            crop_bgr = car_crop[py1_pad:py2_pad, px1_pad:px2_pad]
            if crop_bgr.size == 0:
                continue

            crop_resized = cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_AREA)

            abs_x1 = vx1_c + px1
            abs_y1 = vy1_c + py1
            abs_x2 = vx1_c + px2
            abs_y2 = vy1_c + py2

            pred_text = ocr_predict(lpr_model, crop_resized, device=device)

            crops_found.append({
                "crop_img": crop_resized,
                "conf": conf_val,
                "abs_box": (abs_x1, abs_y1, abs_x2, abs_y2),
                "crop_size": (pw, ph),
                "pred_text": pred_text
            })

    return crops_found


def process_video_cropping(
    video_path,
    output_dir,
    veh_model,
    plate_model,
    lpr_model,
    frame_step=3,
    conf_vehicle=0.30,
    conf_plate=0.18,
    padding_ratio=0.08,
    target_size=(94, 24),
    device="cpu"
):
    video_p = Path(video_path)
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_p}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎬 Đang trích xuất biển số từ video: {video_p.name} ({total_frames} frames, bước nhảy={frame_step})")

    meta_entries = []
    label_entries = []
    frame_idx = 0
    cropped_count = 0

    pbar = tqdm(total=total_frames, desc=f"Cropping {video_p.stem}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if frame_idx % frame_step != 0:
            continue

        crops = crop_plates_from_frame(
            frame, veh_model, plate_model, lpr_model,
            conf_vehicle=conf_vehicle, conf_plate=conf_plate,
            padding_ratio=padding_ratio, target_size=target_size, device=device
        )

        for c_idx, c in enumerate(crops):
            cropped_count += 1
            conf_pct = int(c["conf"] * 100)
            text_str = c["pred_text"] if c["pred_text"] else "UNKNOWN"
            crop_name = f"{video_p.stem}_f{frame_idx:04d}_p{c_idx}_{conf_pct}p_{text_str}.jpg"
            save_path = output_dir / crop_name
            cv2.imwrite(str(save_path), c["crop_img"])

            x1, y1, x2, y2 = c["abs_box"]
            pw, ph = c["crop_size"]
            meta_entries.append(
                f"{crop_name},{video_p.name},{frame_idx},{c['conf']:.3f},{x1},{y1},{x2},{y2},{pw},{ph},{text_str}\n"
            )
            label_entries.append(f"{crop_name} {text_str}\n")

    cap.release()
    pbar.close()
    print(f"✓ Video {video_p.name}: Đã trích xuất {cropped_count} ảnh biển số.")
    return meta_entries, label_entries


def process_image_folder_cropping(
    folder_path,
    output_dir,
    veh_model,
    plate_model,
    lpr_model,
    conf_vehicle=0.30,
    conf_plate=0.18,
    padding_ratio=0.08,
    target_size=(94, 24),
    device="cpu"
):
    folder_p = Path(folder_path)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(str(folder_p / "**" / ext), recursive=True))

    if not image_paths:
        return [], []

    image_paths = sorted(image_paths)
    print(f"\n🖼️  Đang quét qua {len(image_paths):,} khung hình có sẵn trong: {folder_p.resolve()}")

    meta_entries = []
    label_entries = []
    cropped_count = 0

    for idx, img_path_str in enumerate(tqdm(image_paths, desc="Cropping Images")):
        img_p = Path(img_path_str)
        img_bgr = cv2.imread(img_path_str)
        if img_bgr is None:
            continue

        crops = crop_plates_from_frame(
            img_bgr, veh_model, plate_model, lpr_model,
            conf_vehicle=conf_vehicle, conf_plate=conf_plate,
            padding_ratio=padding_ratio, target_size=target_size, device=device
        )

        clip_name = img_p.parent.name
        for c_idx, c in enumerate(crops):
            cropped_count += 1
            conf_pct = int(c["conf"] * 100)
            text_str = c["pred_text"] if c["pred_text"] else "UNKNOWN"
            crop_name = f"{clip_name}_{img_p.stem}_p{c_idx}_{conf_pct}p_{text_str}.jpg"
            save_path = output_dir / crop_name
            cv2.imwrite(str(save_path), c["crop_img"])

            x1, y1, x2, y2 = c["abs_box"]
            pw, ph = c["crop_size"]
            meta_entries.append(
                f"{crop_name},{img_p.name},{idx},{c['conf']:.3f},{x1},{y1},{x2},{y2},{pw},{ph},{text_str}\n"
            )
            label_entries.append(f"{crop_name} {text_str}\n")

    print(f"✓ Thư mục {folder_p.name}: Đã trích xuất {cropped_count} ảnh biển số từ chuỗi frame có sẵn.")
    return meta_entries, label_entries


def main():
    parser = argparse.ArgumentParser(description="Giai đoạn 1: Trích xuất Biển số thực tế (Hỗ trợ cả Video và Frame ảnh cũ)")
    parser.add_argument("--source", type=str, default="data_source", help="Đường dẫn file video, thư mục video hoặc thư mục chuỗi frame ảnh")
    parser.add_argument("--output_dir", type=str, default="data_source/real_cropped_plates", help="Thư mục lưu ảnh cắt")
    parser.add_argument("--vehicle_weights", type=str, default="yolo11n.pt", help="Mô hình phát hiện xe")
    parser.add_argument("--plate_weights", type=str, default="runs/train/yolo11n_license_plate/weights/best.pt", help="Mô hình phát hiện biển số")
    parser.add_argument("--lpr_weights", type=str, default="runs/train/lprnet_bounded_stn/weights/best.pt", help="Mô hình STN+LPRNet đọc chữ")
    parser.add_argument("--step", type=int, default=3, help="Bước nhảy frame đối với video")
    parser.add_argument("--conf_vehicle", type=float, default=0.30, help="Ngưỡng tự tin phát hiện xe")
    parser.add_argument("--conf_plate", type=float, default=0.18, help="Ngưỡng tự tin phát hiện biển số")
    parser.add_argument("--padding", type=float, default=0.08, help="Tỷ lệ nới rộng viền an toàn (padding)")
    args = parser.parse_args()

    source_p = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 1: TRÍCH XUẤT BIỂN SỐ THỰC TẾ (REAL PLATE CROPPING)")
    print(f"📂 Nguồn đầu vào:       {source_p.resolve()}")
    print(f"📁 Thư mục lưu trữ:     {out_dir.resolve()}")
    print(f"🚗 Model phát hiện xe:  {args.vehicle_weights}")
    print(f"🏷️  Model dò biển số:   {args.plate_weights}")
    print(f"🔤 Model đọc chữ OCR:   {args.lpr_weights}")
    print(f"⚙️ Bước nhảy frame:     {args.step} | Smart Padding: {int(args.padding * 100)}%")
    print("=" * 65)

    veh_model = YOLO(args.vehicle_weights)
    plate_model = YOLO(args.plate_weights)
    lpr_model = load_lpr_model(args.lpr_weights, device=device)

    all_meta = ["crop_name,source,frame_idx,conf,xmin,ymin,xmax,ymax,width,height,pseudo_label\n"]
    all_labels = []

    # 1. Nếu nguồn là 1 file video đơn
    if source_p.is_file() and source_p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        m, l = process_video_cropping(
            source_p, out_dir, veh_model, plate_model, lpr_model,
            frame_step=args.step, conf_vehicle=args.conf_vehicle,
            conf_plate=args.conf_plate, padding_ratio=args.padding, device=device
        )
        all_meta.extend(m)
        all_labels.extend(l)

    # 2. Nếu nguồn là thư mục
    elif source_p.is_dir():
        # Kiểm tra xem có video trong thư mục không
        video_files = sorted([p for p in source_p.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        # Kiểm tra cả thư mục con videos/ nếu có
        if not video_files and (source_p / "videos").exists():
            video_files = sorted([p for p in (source_p / "videos").iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])

        if video_files:
            print(f"🎬 Tìm thấy {len(video_files)} video. Bắt đầu trích xuất video...")
            for vf in video_files:
                m, l = process_video_cropping(
                    vf, out_dir, veh_model, plate_model, lpr_model,
                    frame_step=args.step, conf_vehicle=args.conf_vehicle,
                    conf_plate=args.conf_plate, padding_ratio=args.padding, device=device
                )
                all_meta.extend(m)
                all_labels.extend(l)

        # Kiểm tra xem có chuỗi frame ảnh có sẵn không (organized_frames_dataset hoặc ảnh jpg trong thư mục)
        img_check = glob.glob(str(source_p / "**" / "*.jpg"), recursive=True)
        if img_check:
            # Nếu truyền cả thư mục lớn data_source, chỉ quét organized_frames_dataset để tránh quét trùng ảnh đã crop
            target_img_dir = source_p / "organized_frames_dataset" if (source_p / "organized_frames_dataset").exists() else source_p
            if target_img_dir != out_dir:
                m, l = process_image_folder_cropping(
                    target_img_dir, out_dir, veh_model, plate_model, lpr_model,
                    conf_vehicle=args.conf_vehicle, conf_plate=args.conf_plate,
                    padding_ratio=args.padding, device=device
                )
                all_meta.extend(m)
                all_labels.extend(l)

    # Lưu metadata.csv và labels.txt
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.writelines(all_meta)

    label_path = out_dir / "labels.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(all_labels)

    print("\n" + "=" * 65)
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 1: TRÍCH XUẤT BIỂN SỐ THỰC TẾ THÀNH CÔNG!")
    print(f"📊 Tổng số biển số thực tế đã cắt: {len(all_labels):,} ảnh")
    print(f"📁 Thư mục lưu trữ:                 {out_dir.resolve()}")
    print(f"📄 File Metadata:                   {meta_path.resolve()}")
    print(f"🏷️  File Nhãn OCR:                  {label_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
