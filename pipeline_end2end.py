"""
==============================================================================
Bước 4: Pipeline End-to-End ALPR Hoàn Chỉnh (Edge ALPR System)
Kết nối đồng bộ 3 Module lõi theo kiến trúc 2 giai đoạn (Cascaded ALPR):
1. Module Track xe:   YOLO11n + ByteTrack (Nhận diện xe & gán Track ID)
2. Module Dò biển:    YOLO11n Plate Detector (Tìm biển số TRONG THÂN XE)
3. Module Đọc biển:   STN + LPRNet Fine-tuned (Tự nắn thẳng chống rung & đọc chuỗi ký tự)
+ Cơ chế Majority Voting qua Track ID loại bỏ hoàn toàn nhiễu mờ do camera rung!
==============================================================================
"""

import os
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch

from ultralytics import YOLO
from model_stn_lprnet import STN_LPRNet, CHARS, ctc_decode


# ==============================================================================
# HÀM HỖ TRỢ ĐỊNH DẠNG BIỂN SỐ VIỆT NAM (FORMATTING)
# ==============================================================================
def format_vietnamese_plate(raw_text):
    """
    Định dạng chuỗi đọc được thành format biển số Việt Nam chuẩn:
    - 30A12345  -> 30A-123.45
    - 29G112345 -> 29G1-123.45
    - TM1234    -> TM-12.34
    """
    s = raw_text.strip().upper()
    if len(s) == 8:
        return f"{s[:3]}-{s[3:6]}.{s[6:]}"
    elif len(s) == 9:
        return f"{s[:4]}-{s[4:7]}.{s[7:]}"
    elif len(s) == 6:
        return f"{s[:2]}-{s[2:4]}.{s[4:]}"
    elif len(s) == 7:
        return f"{s[:2]}-{s[2:5]}.{s[5:]}"
    return s


# ==============================================================================
# PIPELINE END-TO-END CHÍNH
# ==============================================================================
class EdgeALPRPipeline:
    def __init__(
        self,
        vehicle_weights="yolo11n.pt",
        plate_weights="runs/train/yolo11n_license_plate/weights/best.pt",
        lpr_weights="runs/train/lprnet_finetuned/weights/best.pt",
        tracker_cfg="bytetrack.yaml",
        conf_vehicle=0.30,
        conf_plate=0.18,
        padding_plate=0.08,
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.conf_vehicle = conf_vehicle
        self.conf_plate = conf_plate
        self.padding_plate = padding_plate
        self.tracker_cfg = tracker_cfg

        print("=" * 65)
        print("🚀 ĐANG KHỞI TẠO PIPELINE END-TO-END ALPR (HỆ THỐNG BIÊN)")
        print(f"🖥️ Thiết bị tính toán:       {self.device}")

        # 1. Tải Model Track xe (COCO)
        print(f"🚗 Module 1: Track xe      -> {vehicle_weights}")
        self.vehicle_model = YOLO(vehicle_weights)

        # 2. Tải Model Dò biển số
        print(f"🏷️  Module 2: Dò biển số     -> {plate_weights}")
        self.plate_model = YOLO(plate_weights)

        # 3. Tải Model STN + LPRNet Đọc ký tự
        print(f"🔤 Module 3: Đọc chữ (OCR)  -> {lpr_weights}")
        class_num = len(CHARS) + 1
        self.lpr_model = STN_LPRNet(class_num=class_num, in_channels=3, use_stn=True)
        
        if Path(lpr_weights).exists():
            ckpt = torch.load(lpr_weights, map_location="cpu")
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.lpr_model.load_state_dict(state_dict)
            print("   ✓ Đã nạp thành công trọng số LPRNet + STN!")
        else:
            print(f"   ⚠️ Không tìm thấy trọng số OCR: {lpr_weights}, dùng trọng số khởi tạo ngẫu nhiên.")
        
        self.lpr_model.to(self.device)
        self.lpr_model.eval()

        # Bộ nhớ theo dõi đa khung hình (Temporal Voting Memory)
        self.track_history = defaultdict(list)
        self.final_plate_decision = {}

        # Class xe cần theo dõi từ COCO:
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        print("=" * 65)

    def read_plate_ocr(self, plate_crop_bgr):
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return ""

        img_resized = cv2.resize(plate_crop_bgr, (94, 24), interpolation=cv2.INTER_AREA)
        img_norm = (img_resized.astype(np.float32) - 127.5) * 0.0078125
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.lpr_model(img_tensor)
            pred_text = ctc_decode(logits, CHARS, blank_idx=0)[0]

        return pred_text

    def process_frame(self, frame_bgr):
        h_frame, w_frame = frame_bgr.shape[:2]
        annotated_frame = frame_bgr.copy()

        # -------------------------------------------------------------
        # MODULE 1: BÁM VẾT PHƯƠNG TIỆN (BYTE TRACK)
        # -------------------------------------------------------------
        veh_results = self.vehicle_model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker_cfg,
            classes=list(self.vehicle_classes.keys()),
            conf=self.conf_vehicle,
            verbose=False
        )

        current_vehicles = []
        if veh_results and len(veh_results) > 0 and veh_results[0].boxes is not None:
            boxes = veh_results[0].boxes
            xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else [-1] * len(boxes)
            )

            for (vx1, vy1, vx2, vy2), cls_id, track_id in zip(xyxy_arr, cls_arr, track_ids):
                if (vx2 - vx1) < 50 or (vy2 - vy1) < 40:
                    continue

                vx1_c = max(0, vx1)
                vy1_c = max(0, vy1)
                vx2_c = min(w_frame, vx2)
                vy2_c = min(h_frame, vy2)

                current_vehicles.append({
                    "track_id": track_id,
                    "cls_name": self.vehicle_classes.get(cls_id, "Vehicle"),
                    "box": (vx1_c, vy1_c, vx2_c, vy2_c)
                })

        # -------------------------------------------------------------
        # MODULE 2 & 3: DÒ BIỂN SỐ TRONG THÂN XE & ĐỌC CHỮ (OCR)
        # -------------------------------------------------------------
        for veh in current_vehicles:
            vx1, vy1, vx2, vy2 = veh["box"]
            track_id = veh["track_id"]
            cls_name = veh["cls_name"]

            car_crop = frame_bgr[vy1:vy2, vx1:vx2]
            if car_crop.size == 0:
                continue

            plate_results = self.plate_model(car_crop, conf=self.conf_plate, verbose=False)

            detected_plate_str = ""
            best_plate_box_frame = None

            if plate_results and len(plate_results) > 0 and plate_results[0].boxes is not None:
                p_boxes = plate_results[0].boxes
                if len(p_boxes) > 0:
                    best_idx = torch.argmax(p_boxes.conf).item()
                    p_xyxy = p_boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    px1_car, py1_car, px2_car, py2_car = p_xyxy

                    pw = px2_car - px1_car
                    ph = py2_car - py1_car

                    pad_w = int(pw * self.padding_plate)
                    pad_h = int(ph * self.padding_plate)

                    px1_pad = max(0, px1_car - pad_w)
                    py1_pad = max(0, py1_car - pad_h)
                    px2_pad = min(car_crop.shape[1], px2_car + pad_w)
                    py2_pad = min(car_crop.shape[0], py2_car + pad_h)

                    plate_crop = car_crop[py1_pad:py2_pad, px1_pad:px2_pad]

                    best_plate_box_frame = (
                        vx1 + px1_car,
                        vy1 + py1_car,
                        vx1 + px2_car,
                        vy1 + py2_car
                    )

                    raw_text = self.read_plate_ocr(plate_crop)
                    if len(raw_text) >= 2:
                        detected_plate_str = raw_text

            # -------------------------------------------------------------
            # THUẬT TOÁN MAJORITY VOTING (BẦU CHỌN ĐA SỐ CHỐNG RUNG LẮC)
            # -------------------------------------------------------------
            if detected_plate_str:
                self.track_history[track_id].append(detected_plate_str)
                if len(self.track_history[track_id]) > 20:
                    self.track_history[track_id].pop(0)

                most_common_plate, count = Counter(self.track_history[track_id]).most_common(1)[0]
                self.final_plate_decision[track_id] = most_common_plate

            display_plate = self.final_plate_decision.get(track_id, "")
            formatted_plate = format_vietnamese_plate(display_plate) if display_plate else ""

            # -------------------------------------------------------------
            # VẼ TRỰC QUAN HÓA (VISUALIZATION)
            # -------------------------------------------------------------
            # 1. Vẽ Box xe
            veh_color = (0, 230, 115)  # Xanh ngọc
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), veh_color, 2)

            veh_label = f"ID:{track_id} {cls_name}"
            (tw, th), _ = cv2.getTextSize(veh_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (vx1, vy1 - th - 8), (vx1 + tw + 6, vy1), veh_color, -1)
            cv2.putText(annotated_frame, veh_label, (vx1 + 3, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 2. Vẽ Box biển số
            if best_plate_box_frame is not None:
                px1, py1, px2, py2 = best_plate_box_frame
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 100, 255), 2)

            # 3. Vẽ Banner hiển thị Biển số xe
            if formatted_plate:
                plate_text = f"[{formatted_plate}]"
                font_scale = 0.75
                thickness = 2
                (ptw, pth), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                banner_x = vx1 + 5
                banner_y = max(pth + 10, vy1 - th - 12)

                cv2.rectangle(annotated_frame, (banner_x - 4, banner_y - pth - 6), (banner_x + ptw + 6, banner_y + 4), (20, 20, 20), -1)
                cv2.rectangle(annotated_frame, (banner_x - 4, banner_y - pth - 6), (banner_x + ptw + 6, banner_y + 4), (0, 220, 255), 2)
                cv2.putText(annotated_frame, plate_text, (banner_x, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        return annotated_frame


def process_single_video_pipeline(video_path, out_video_path, snap_dir, pipeline):
    video_p = Path(video_path)
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_p}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_p = Path(out_video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    snap_p = Path(snap_dir) if snap_dir else None
    if snap_p:
        snap_p.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, input_fps, (w, h))

    print(f"\n🎬 Đang xử lý Video: {video_p.name} ({total_frames} frames, {w}x{h} px, {input_fps:.1f} FPS)")
    frame_idx = 0
    t_start = time.time()
    saved_snaps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        annotated = pipeline.process_frame(frame)
        proc_fps = 1.0 / max(1e-5, (time.time() - t0))

        # Lưu ảnh chụp nhanh (snapshot) nếu có biển số phát hiện
        active_plates = [p for p in pipeline.final_plate_decision.values() if p]
        if snap_p and active_plates and frame_idx % 15 == 0 and saved_snaps < 8:
            snap_file = snap_p / f"{video_p.stem}_f{frame_idx:04d}_{'_'.join(active_plates[:2])}.jpg"
            cv2.imwrite(str(snap_file), annotated)
            saved_snaps += 1

        cv2.putText(annotated, f"Edge ALPR | FPS: {proc_fps:.1f} | Frame: {frame_idx}/{total_frames}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 60 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            print(f"   -> Frame {frame_idx:04d}/{total_frames} | Tốc độ: {frame_idx/elapsed:.1f} FPS | Đang nhận diện {len(pipeline.final_plate_decision)} xe")

    cap.release()
    writer.release()
    total_time = time.time() - t_start

    print(f"\n--- KẾT QUẢ CHO {video_p.name} ---")
    print(f"⏱️ Tổng thời gian:     {total_time:.2f}s ({total_frames/total_time:.1f} FPS)")
    print(f"🚗 Tổng số xe theo dõi: {len(pipeline.track_history)} xe")
    print(f"🏷️ Danh sách biển số nhận diện theo Track ID:")
    for track_id, plate in pipeline.final_plate_decision.items():
        history = pipeline.track_history.get(track_id, [])
        counts = Counter(history)
        print(f"   - [Track ID #{track_id}] Biển số: {format_vietnamese_plate(plate)} (Tần suất: {counts.get(plate, 0)}/{len(history)} frames)")
    print(f"📁 Video xuất: {out_p.resolve()}")


def run_pipeline(
    source="data_source/videos",
    output_path="output/final_end2end",
    vehicle_model="yolo11n.pt",
    plate_model="runs/train/yolo11n_license_plate/weights/best.pt",
    lpr_model="runs/train/lprnet_finetuned/weights/best.pt",
    conf_vehicle=0.30,
    conf_plate=0.18
):
    source_p = Path(source)
    out_p = Path(output_path)
    out_p.mkdir(parents=True, exist_ok=True)
    snap_dir = out_p / "snapshots"

    # 1. Nếu nguồn là 1 video đơn
    if source_p.is_file() and source_p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        pipeline = EdgeALPRPipeline(
            vehicle_weights=vehicle_model,
            plate_weights=plate_model,
            lpr_weights=lpr_model,
            conf_vehicle=conf_vehicle,
            conf_plate=conf_plate
        )
        out_vid = out_p / f"{source_p.stem}_annotated.mp4" if out_p.is_dir() else out_p
        process_single_video_pipeline(source_p, out_vid, snap_dir, pipeline)

    # 2. Nếu nguồn là thư mục chứa nhiều video
    elif source_p.is_dir():
        video_files = sorted([p for p in source_p.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        if not video_files and (source_p / "videos").exists():
            video_files = sorted([p for p in (source_p / "videos").iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])

        if video_files:
            print(f"🎬 Tìm thấy {len(video_files)} video. Bắt đầu xử lý từng video...")
            for vf in video_files:
                pipeline = EdgeALPRPipeline(
                    vehicle_weights=vehicle_model,
                    plate_weights=plate_model,
                    lpr_weights=lpr_model,
                    conf_vehicle=conf_vehicle,
                    conf_plate=conf_plate
                )
                out_vid = out_p / f"{vf.stem}_annotated.mp4"
                process_single_video_pipeline(vf, out_vid, snap_dir, pipeline)
        else:
            # Xử lý chuỗi frame ảnh nếu không có video
            image_files = sorted(list(source_p.glob("*.jpg")) + list(source_p.glob("*.png")))
            if image_files:
                pipeline = EdgeALPRPipeline(
                    vehicle_weights=vehicle_model,
                    plate_weights=plate_model,
                    lpr_weights=lpr_model,
                    conf_vehicle=conf_vehicle,
                    conf_plate=conf_plate
                )
                out_dir = out_p / f"pipeline_{source_p.name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                sample_img = cv2.imread(str(image_files[0]))
                h, w = sample_img.shape[:2]
                video_out_path = out_p / f"{source_p.name}_annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out_path), fourcc, 10.0, (w, h))

                for idx, img_file in enumerate(image_files):
                    frame = cv2.imread(str(img_file))
                    if frame is None:
                        continue
                    annotated = pipeline.process_frame(frame)
                    cv2.imwrite(str(out_dir / img_file.name), annotated)
                    writer.write(annotated)
                writer.release()
                print(f"✓ Hoàn tất chuỗi frame: {video_out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline End-to-End ALPR nhận diện xe và biển số")
    parser.add_argument("--source", type=str, default="data_source/videos", help="Đường dẫn video / thư mục video")
    parser.add_argument("--output", type=str, default="output/final_end2end", help="Thư mục lưu video và snapshot kết quả")
    parser.add_argument("--vehicle_model", type=str, default="yolo11n.pt", help="Mô hình phát hiện xe")
    parser.add_argument("--plate_model", type=str, default="runs/train/yolo11n_license_plate/weights/best.pt", help="Mô hình dò biển số")
    parser.add_argument("--lpr_model", type=str, default="runs/train/lprnet_finetuned/weights/best.pt", help="Mô hình STN + LPRNet fine-tuned")
    parser.add_argument("--conf_vehicle", type=float, default=0.30, help="Ngưỡng tự tin phát hiện xe")
    parser.add_argument("--conf_plate", type=float, default=0.18, help="Ngưỡng tự tin phát hiện biển số")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output_path=args.output,
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        lpr_model=args.lpr_model,
        conf_vehicle=args.conf_vehicle,
        conf_plate=args.conf_plate
    )
