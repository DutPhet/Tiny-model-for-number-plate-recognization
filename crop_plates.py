"""
==============================================================================
Script 1: Real License Plate Cropper (Trích xuất Biển số Thật bằng YOLO11n)
- Quét qua toàn bộ frame ảnh từ camera hành trình hoặc dataset gốc.
- Dùng mô hình YOLO11n phát hiện biển số (best.pt) để xác định vị trí.
- Cơ chế Smart Padding (nới viền an toàn 8%) tránh mất nét chữ khi xe bị rung.
- Tự động chuẩn hóa kích thước về chuẩn LPRNet (94x24 px).
- Xuất toàn bộ ảnh cắt và file metadata.csv phục vụ kiểm thử và tinh chỉnh.
==============================================================================
"""

import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def crop_plates_from_dataset(
    dataset_dir="data_source/organized_frames_dataset",
    output_dir="data_source/real_cropped_plates",
    model_path="runs/train/yolo11n_license_plate/weights/best.pt",
    conf_thresh=0.25,
    padding_ratio=0.08,
    target_size=(94, 24),
    save_raw=False
):
    """
    Trích xuất toàn bộ biển số xe từ tập dữ liệu thực tế bằng YOLO11n.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra mô hình YOLO11n
    model_file = Path(model_path)
    if not model_file.exists():
        fallback_model = Path("yolo11n.pt")
        if fallback_model.exists():
            print(f"⚠️ Không tìm thấy {model_file}, chuyển sang dùng {fallback_model}")
            model_file = fallback_model
        else:
            print(f"❌ Không tìm thấy mô hình: {model_file}")
            return

    print("=" * 65)
    print("🚗🔍 SCRIPT 1: TRÍCH XUẤT BIỂN SỐ THỰC TẾ BẰNG YOLO11n")
    print(f"📂 Thư mục nguồn:       {dataset_dir.resolve()}")
    print(f"📁 Thư mục lưu crops:   {output_dir.resolve()}")
    print(f"🎯 Model YOLO:          {model_file}")
    print(f"⚙️ Ngưỡng Conf:         {conf_thresh} | Smart Padding: {int(padding_ratio * 100)}%")
    print(f"📐 Kích thước chuẩn:    {target_size[0]} x {target_size[1]} px")
    print("=" * 65)

    model = YOLO(str(model_file))

    # 2. Tìm kiếm toàn bộ ảnh trong thư mục nguồn (hỗ trợ cả lồng thư mục con/clip)
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(str(dataset_dir / "**" / ext), recursive=True))

    if not image_paths:
        print(f"❌ Không tìm thấy ảnh nào trong: {dataset_dir}")
        return

    # Sắp xếp ảnh
    image_paths = sorted(image_paths)
    print(f"✓ Đã tìm thấy {len(image_paths):,} ảnh. Bắt đầu trích xuất...\n")

    metadata_lines = ["crop_name,source_image,clip_name,conf,xmin,ymin,xmax,ymax,crop_w,crop_h\n"]
    total_cropped = 0
    tw, th = target_size

    # 3. Quét và trích xuất
    for img_path_str in tqdm(image_paths, desc="Processing Images"):
        img_p = Path(img_path_str)
        img_bgr = cv2.imread(img_path_str)
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]
        clip_name = img_p.parent.name

        # Dự đoán bằng YOLO11n
        results = model.predict(
            source=img_bgr,
            conf=conf_thresh,
            verbose=False
        )

        if not results or len(results) == 0:
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for b_idx, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = xyxy

            bw = x2 - x1
            bh = y2 - y1

            # Bỏ qua các box quá nhỏ hoặc không hợp lệ (nhiễu)
            if bw < 10 or bh < 6:
                continue

            # Áp dụng Smart Padding: Nới rộng viền an toàn quanh biển số
            pad_w = bw * padding_ratio
            pad_h = bh * padding_ratio

            x1_pad = max(0, int(x1 - pad_w))
            y1_pad = max(0, int(y1 - pad_h))
            x2_pad = min(w_img, int(x2 + pad_w))
            y2_pad = min(h_img, int(y2 + pad_h))

            crop_bgr = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop_bgr.size == 0:
                continue

            # Đặt tên file crop có ý nghĩa
            crop_filename = f"{clip_name}_{img_p.stem}_plate{b_idx}_{int(conf * 100)}p.jpg"
            dest_crop_path = output_dir / crop_filename

            # Lưu ảnh
            if save_raw:
                cv2.imwrite(str(dest_crop_path), crop_bgr)
            else:
                # Resize về chuẩn LPRNet (94x24 px)
                crop_resized = cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dest_crop_path), crop_resized)

            total_cropped += 1
            metadata_lines.append(
                f"{crop_filename},{img_p.name},{clip_name},{conf:.3f},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{int(bw)},{int(bh)}\n"
            )

    # 4. Lưu metadata
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.writelines(metadata_lines)

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT TRÍCH XUẤT BIỂN SỐ THỰC TẾ!")
    print(f"📊 Tổng số ảnh đã quét:      {len(image_paths):,}")
    print(f"🏷️  Tổng số biển số cắt được: {total_cropped:,}")
    print(f"📁 Thư mục lưu:              {output_dir.resolve()}")
    print(f"📄 Bảng thống kê chi tiết:   {meta_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script 1: Crop biển số xe thực tế bằng YOLO11n")
    parser.add_argument("--dataset_dir", type=str, default="data_source/organized_frames_dataset", help="Thư mục ảnh gốc")
    parser.add_argument("--output_dir", type=str, default="data_source/real_cropped_plates", help="Thư mục lưu ảnh cắt")
    parser.add_argument("--model", type=str, default="runs/train/yolo11n_license_plate/weights/best.pt", help="Trọng số YOLO11n")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tự tin phát hiện")
    parser.add_argument("--padding", type=float, default=0.08, help="Tỷ lệ nới rộng viền an toàn (padding)")
    parser.add_argument("--width", type=int, default=94, help="Chiều rộng ảnh chuẩn hóa LPRNet")
    parser.add_argument("--height", type=int, default=24, help="Chiều cao ảnh chuẩn hóa LPRNet")
    parser.add_argument("--save_raw", action="store_true", help="Lưu độ phân giải gốc thay vì resize 94x24")
    args = parser.parse_args()

    crop_plates_from_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_thresh=args.conf,
        padding_ratio=args.padding,
        target_size=(args.width, args.height),
        save_raw=args.save_raw
    )
