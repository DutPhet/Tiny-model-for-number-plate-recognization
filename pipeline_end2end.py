"""
==============================================================================
Bước 4: Pipeline End-to-End ALPR Hoàn Chỉnh (Edge ALPR System)
Kết nối đồng bộ 3 Module lõi theo kiến trúc 2 giai đoạn (Cascaded ALPR):
1. Module Track xe:   YOLO11n + ByteTrack (Nhận diện xe & gán Track ID)
2. Module Dò biển:    YOLO11n Plate Detector (Tìm biển số TRONG THÂN XE)
3. Module Đọc biển:   STN + LPRNet (Tự nắn thẳng chống rung & đọc chuỗi ký tự)
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
        # Ví dụ: 30A12345 -> 30A - 123.45
        return f"{s[:3]}-{s[3:6]}.{s[6:]}"
    elif len(s) == 9:
        # Ví dụ: 29G112345 -> 29G1 - 123.45
        return f"{s[:4]}-{s[4:7]}.{s[7:]}"
    elif len(s) == 6:
        # Ví dụ xe quân đội 4 số: TM1234 -> TM - 12.34
        return f"{s[:2]}-{s[2:4]}.{s[4:]}"
    elif len(s) == 7:
        # Xe biển 4 số cũ hoặc quân đội 5 số: QK12345 -> QK - 123.45
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
        lpr_weights="runs/train/lprnet_bounded_stn/weights/best.pt",
        tracker_cfg="bytetrack.yaml",
        conf_vehicle=0.35,
        conf_plate=0.25,
        padding_plate=0.08,
        device=None
    ):
        self.conf_vehicle = conf_vehicle
        self.conf_plate = conf_plate
        self.padding_plate = padding_plate
        self.tracker_cfg = tracker_cfg

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
            print(f"   ⚠️ CẢNH BÁO: Không tìm thấy {lpr_weights}, dùng trọng số ngẫu nhiên.")

        self.lpr_model.to(self.device)
        self.lpr_model.eval()

        # Bộ nhớ theo dõi đa frame để bầu chọn (Majority Voting Buffer)
        # track_history[track_id] = list các kết quả đọc qua từng frame
        self.track_history = defaultdict(list)
        self.final_plate_decision = {}

        # Danh sách tên class xe COCO
        self.vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        print("=" * 65)

    def read_plate_ocr(self, plate_crop_bgr):
        """
        Nắn thẳng bằng STN và giải mã chuỗi ký tự bằng LPRNet:
        - Input:  Ảnh cắt ô biển số (BGR)
        - Output: Chuỗi text biển số đọc được
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return ""

        # Resize chuẩn kích thước (94, 24) và chuẩn hóa [-1, 1]
        img_resized = cv2.resize(plate_crop_bgr, (94, 24), interpolation=cv2.INTER_AREA)
        img_norm = (img_resized.astype(np.float32) - 127.5) * 0.0078125
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, rectified_img, theta = self.lpr_model(img_tensor, return_rectified=True)
            pred_text = ctc_decode(logits, CHARS, blank_idx=0)[0]

        return pred_text

    def process_frame(self, frame_bgr):
        """
        Xử lý 1 khung hình qua trọn vẹn 3 Module:
        1. Track xe -> Bounding box xe + Track ID
        2. Dò biển số BÊN TRONG THÂN XE -> Bounding box biển
        3. Đọc biển số -> Nắn thẳng STN + OCR LPRNet + Majority Vote
        4. Vẽ kết quả trực quan
        """
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
                # Bỏ qua xe quá nhỏ ở đường chân trời xa
                if (vx2 - vx1) < 60 or (vy2 - vy1) < 50:
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

            # Cắt ảnh thân xe (Loại bỏ hoàn toàn góc ngày giờ ở đáy video!)
            car_crop = frame_bgr[vy1:vy2, vx1:vx2]
            if car_crop.size == 0:
                continue

            # Chạy dò biển số bên trong ô thân xe
            plate_results = self.plate_model(car_crop, conf=self.conf_plate, verbose=False)

            detected_plate_str = ""
            best_plate_box_frame = None

            if plate_results and len(plate_results) > 0 and plate_results[0].boxes is not None:
                p_boxes = plate_results[0].boxes
                if len(p_boxes) > 0:
                    # Lấy biển số có độ tin cậy cao nhất trong xe
                    best_idx = torch.argmax(p_boxes.conf).item()
                    p_xyxy = p_boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    px1_car, py1_car, px2_car, py2_car = p_xyxy

                    pw = px2_car - px1_car
                    ph = py2_car - py1_car

                    # Nới rộng viền thông minh (Smart Padding 8%)
                    pad_w = int(pw * self.padding_plate)
                    pad_h = int(ph * self.padding_plate)

                    px1_pad = max(0, px1_car - pad_w)
                    py1_pad = max(0, py1_car - pad_h)
                    px2_pad = min(car_crop.shape[1], px2_car + pad_w)
                    py2_pad = min(car_crop.shape[0], py2_car + pad_h)

                    plate_crop = car_crop[py1_pad:py2_pad, px1_pad:px2_pad]

                    # Chuyển đổi tọa độ biển số sang hệ tọa độ toàn khung hình
                    best_plate_box_frame = (
                        vx1 + px1_car,
                        vy1 + py1_car,
                        vx1 + px2_car,
                        vy1 + py2_car
                    )

                    # ĐỌC CHỮ BẰNG STN + LPRNET
                    raw_text = self.read_plate_ocr(plate_crop)
                    if len(raw_text) >= 4:
                        detected_plate_str = raw_text

            # -------------------------------------------------------------
            # THUẬT TOÁN MAJORITY VOTING (BẦU CHỌN ĐA SỐ CHỐNG RUNG LẮC)
            # -------------------------------------------------------------
            if detected_plate_str:
                self.track_history[track_id].append(detected_plate_str)
                # Giữ tối đa 20 frame gần nhất
                if len(self.track_history[track_id]) > 20:
                    self.track_history[track_id].pop(0)

                # Chọn kết quả xuất hiện nhiều nhất (Majority Vote)
                most_common_plate, count = Counter(self.track_history[track_id]).most_common(1)[0]
                self.final_plate_decision[track_id] = most_common_plate

            display_plate = self.final_plate_decision.get(track_id, "")
            display_plate_formatted = format_vietnamese_plate(display_plate) if display_plate else ""

            # -------------------------------------------------------------
            # VẼ KẾT QUẢ TRỰC QUAN LÊN KHUNG HÌNH (VISUALIZATION)
            # -------------------------------------------------------------
            # 1. Vẽ Box xe (Màu xanh dương/ngọc)
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 220, 255), 2)

            # 2. Vẽ Box biển số nếu phát hiện được (Màu cam/đỏ)
            if best_plate_box_frame is not None:
                bx1, by1, bx2, by2 = best_plate_box_frame
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 69, 255), 2)

            # 3. Banner thông tin xe + Biển số phía trên nóc xe
            track_label = f"#{track_id} {cls_name}"
            if display_plate_formatted:
                full_banner = f"{track_label} | {display_plate_formatted}"
                banner_color = (0, 200, 0) # Xanh lá đậm khi đã đọc được biển
            else:
                full_banner = track_label
                banner_color = (0, 160, 220)

            # Tạo nền đen mờ sau chữ để chữ luôn nổi bật
            (tw, th), baseline = cv2.getTextSize(full_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            banner_y1 = max(0, vy1 - th - 12)
            banner_y2 = vy1
            cv2.rectangle(annotated_frame, (vx1, banner_y1), (vx1 + tw + 10, banner_y2), (20, 20, 20), -1)
            cv2.putText(
                annotated_frame,
                full_banner,
                (vx1 + 5, banner_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                banner_color,
                2,
                cv2.LINE_AA
            )

        return annotated_frame


# ==============================================================================
# HÀM THỰC THI CHÍNH TRÊN FILE VIDEO HOẶC THƯ MỤC FRAME
# ==============================================================================
def run_pipeline(
    source,
    output_path="output/pipeline_result.mp4",
    vehicle_model="yolo11n.pt",
    plate_model="runs/train/yolo11n_license_plate/weights/best.pt",
    lpr_model="runs/train/lprnet_bounded_stn/weights/best.pt",
    conf_vehicle=0.35,
    conf_plate=0.25,
    show=False
):
    source_p = Path(source)
    pipeline = EdgeALPRPipeline(
        vehicle_weights=vehicle_model,
        plate_weights=plate_model,
        lpr_weights=lpr_model,
        conf_vehicle=conf_vehicle,
        conf_plate=conf_plate
    )

    # 1. Trường hợp Source là một thư mục chứa chuỗi ảnh frame (Clip camera hành trình)
    if source_p.is_dir():
        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        image_files = []
        for ext in image_exts:
            image_files.extend(source_p.glob(ext))
        image_files = sorted(image_files)

        if not image_files:
            print(f"❌ Không tìm thấy ảnh nào trong thư mục: {source_p}")
            return

        print(f"\n🎬 Bắt đầu xử lý chuỗi {len(image_files)} khung hình từ clip: {source_p.name}")
        out_dir = Path("output") / f"pipeline_{source_p.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo VideoWriter để ghép chuỗi frame thành video MP4
        sample_img = cv2.imread(str(image_files[0]))
        h, w = sample_img.shape[:2]
        video_out_path = Path("output") / f"{source_p.name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, 10.0, (w, h))

        t_start = time.time()
        for idx, img_file in enumerate(image_files):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue

            t0 = time.time()
            annotated = pipeline.process_frame(frame)
            fps = 1.0 / max(1e-5, (time.time() - t0))

            # Vẽ chỉ số FPS góc trái trên
            cv2.putText(annotated, f"Edge ALPR Pipeline | FPS: {fps:.1f}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Lưu ảnh kết quả và ghi vào video
            cv2.imwrite(str(out_dir / img_file.name), annotated)
            writer.write(annotated)

            if (idx + 1) % 5 == 0 or idx + 1 == len(image_files):
                print(f"   -> Đã xử lý {idx + 1}/{len(image_files)} frames ({fps:.1f} FPS)")

        writer.release()
        total_time = time.time() - t_start
        print("\n" + "=" * 65)
        print("🎉 HOÀN TẤT XỬ LÝ TOÀN BỘ CHUỖI KHUNG HÌNH!")
        print(f"⏱️ Tổng thời gian:     {total_time:.2f}s (Tốc độ TB: {len(image_files)/total_time:.1f} FPS)")
        print(f"📁 Thư mục ảnh xuất:   {out_dir.resolve()}")
        print(f"🎬 Video xuất:         {video_out_path.resolve()}")
        print("=" * 65)

    # 2. Trường hợp Source là một Video đơn (.mp4, .avi)
    elif source_p.is_file() and source_p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(str(source_p))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, input_fps, (w, h))

        print(f"\n🎬 Bắt đầu xử lý Video: {source_p.name} ({total_frames} frames, {w}x{h} px)")
        frame_idx = 0
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            annotated = pipeline.process_frame(frame)
            fps = 1.0 / max(1e-5, (time.time() - t0))

            cv2.putText(annotated, f"FPS: {fps:.1f}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                print(f"   -> Đang xử lý: {frame_idx}/{total_frames} frames ({fps:.1f} FPS)")

        cap.release()
        writer.release()
        print("\n" + "=" * 65)
        print(f"🎉 HOÀN TẤT XỬ LÝ VIDEO! Đã lưu tại: {out_path.resolve()}")
        print("=" * 65)

    # 3. Trường hợp Source là một file ảnh đơn (.jpg, .png)
    elif source_p.is_file():
        frame = cv2.imread(str(source_p))
        if frame is None:
            print(f"❌ Không đọc được ảnh: {source_p}")
            return
        annotated = pipeline.process_frame(frame)
        out_path = Path("output") / f"annotated_{source_p.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"✓ Đã xử lý ảnh thành công: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline End-to-End ALPR nhận diện xe và biển số")
    parser.add_argument("--source", type=str, default="data_source/organized_frames_dataset/test/1f652a87_447_clip_005_0027_0034_N", help="Đường dẫn video / thư mục clip / ảnh")
    parser.add_argument("--output", type=str, default="output/pipeline_output.mp4", help="Đường dẫn lưu video kết quả")
    parser.add_argument("--vehicle_model", type=str, default="yolo11n.pt", help="Mô hình phát hiện xe")
    parser.add_argument("--plate_model", type=str, default="runs/train/yolo11n_license_plate/weights/best.pt", help="Mô hình dò biển số")
    parser.add_argument("--lpr_model", type=str, default="runs/train/lprnet_bounded_stn/weights/best.pt", help="Mô hình STN + LPRNet đọc chữ")
    parser.add_argument("--conf_vehicle", type=float, default=0.35, help="Ngưỡng tự tin phát hiện xe")
    parser.add_argument("--conf_plate", type=float, default=0.25, help="Ngưỡng tự tin phát hiện biển số")
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
