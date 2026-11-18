import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import auto_find_m1_checkpoint, resolve_project_paths
from module1_worker import Module1RealtimeWorker
from osd_renderer import render_module1_osd
from sensor_stream import DashcamSensorStream


def main():
    parser = argparse.ArgumentParser(description="Demo Realtime Chỉ Chạy Module 1 (Vehicle D-FINE + Hybrid Tracker)")
    parser.add_argument("--source", type=str, default="0", help="'0' cho Webcam, URL RTSP, hoặc đường dẫn file video")
    parser.add_argument("--ckpt", type=str, default="", help="Đường dẫn checkpoint Module 1 (.pth). Bỏ trống để tự tìm")
    parser.add_argument("--conf-thresh", type=float, default=0.45, help="Ngưỡng tin cậy chính phát hiện xe (mặc định: 0.45)")
    parser.add_argument("--low-thresh", type=float, default=0.18, help="Ngưỡng phụ ByteTrack giữ track mờ (mặc định: 0.18)")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ phát video (1.0 = realtime chuẩn)")
    parser.add_argument("--loop", action="store_true", help="Phát lặp lại liên tục nếu nguồn là file video")
    parser.add_argument("--raw-only", action="store_true", help="Khởi động ở chế độ chỉ hiện box thô của Module 1 (tắt Tracker)")
    parser.add_argument("--no-cmc", action="store_true", help="Tắt bù chuyển động camera (CMC Optical Flow)")
    parser.add_argument("--no-norm", action="store_true", help="Tắt chuẩn hóa ImageNet mean/std (chỉ dùng nếu train không norm)")
    parser.add_argument("--fp32", action="store_true", help="Ép chạy FP32 thay vì FP16 trên GPU")
    parser.add_argument("--record", action="store_true", help="Ghi lại video kết quả ra file .mp4")
    parser.add_argument("--output-dir", type=str, default="", help="Thư mục lưu log & video ghi hình")
    parser.add_argument("--no-show", action="store_true", help="Chạy không mở cửa sổ GUI (headless)")
    args = parser.parse_args()

    _, project_root = resolve_project_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = auto_find_m1_checkpoint(args.ckpt)

    out_dir = Path(args.output_dir) if args.output_dir else (project_root / "runs" / "module1_realtime_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor = DashcamSensorStream(
        source=args.source.strip(),
        loop_video=args.loop,
        speed=args.speed,
    )
    worker = Module1RealtimeWorker(
        sensor=sensor,
        ckpt_path=ckpt_path,
        device=device,
        out_dir=out_dir,
        conf_thresh=args.conf_thresh,
        low_thresh=args.low_thresh,
        use_fp16=not args.fp32,
        use_cmc=not args.no_cmc,
        use_imagenet_norm=not args.no_norm,
    )

    writer = None
    if args.record:
        rec_path = out_dir / f"m1_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(
            str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), sensor.fps, (sensor.width, sensor.height)
        )
        print(f"🎥 Đang ghi hình ra: {rec_path}")

    print()
    print("🚘 Module 1 Realtime đang chạy!")
    print("   Phím tắt: [Q] Thoát | [T] Đổi Raw Box <-> Tracker | [C] Bật/Tắt CMC | [S] Chụp ảnh | [R] Reset | [P] Tạm dừng")
    print()

    sensor.start()
    worker.start()

    show_raw_only = args.raw_only
    paused = False
    last_vis = None

    try:
        while True:
            if not paused:
                frame, frame_ts, _ = sensor.get_latest()
                if frame is None:
                    if sensor.eof:
                        break
                    time.sleep(0.005)
                    continue

                tracks, trails, raw_det, counts, total_unique, cmc_on = worker.get_snapshot()
                vis = render_module1_osd(
                    frame_bgr=frame,
                    frame_ts=frame_ts,
                    sensor=sensor,
                    worker=worker,
                    tracks=tracks,
                    trails=trails,
                    raw_detections=raw_det,
                    unique_counts=counts,
                    total_unique=total_unique,
                    cmc_on=cmc_on,
                    show_raw_only=show_raw_only,
                )
                last_vis = vis
                if writer is not None:
                    writer.write(vis)
            else:
                vis = last_vis

            if not args.no_show and vis is not None:
                cv2.imshow("Module 1 Vehicle Realtime Test [Q:Quit | T:Raw/Track | C:CMC | S:Snap | R:Reset]", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("t"), ord("T")):
                    show_raw_only = not show_raw_only
                    print(f"🔀 Chế độ hiển thị: {'RAW DETECTOR (Chỉ D-FINE)' if show_raw_only else 'HYBRID TRACKER (#ID)'}")
                elif key in (ord("c"), ord("C")):
                    state = worker.toggle_cmc()
                    print(f"📷 Camera Motion Compensation (CMC): {'BẬT' if state else 'TẮT'}")
                elif key in (ord("s"), ord("S")):
                    snap_p = out_dir / f"m1_snap_{datetime.now().strftime('%H%M%S_%f')[:10]}.jpg"
                    cv2.imencode(".jpg", vis)[1].tofile(str(snap_p))
                    print(f"📸 Đã lưu ảnh chụp màn hình: {snap_p}")
                elif key in (ord("r"), ord("R")):
                    worker.reset_state()
                    print("🔄 Đã reset bộ đếm và tracker.")
                elif key in (ord("p"), ord("P")):
                    paused = not paused
            else:
                time.sleep(1.0 / sensor.fps)

            if sensor.eof and not paused:
                break
    finally:
        worker.stop()
        sensor.stop()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print()
    print(
        f"🏁 Hoàn tất kiểm tra Module 1 | Chu kỳ AI đã chạy: {worker.processed_cycles} | "
        f"Frame bỏ qua do trễ: {worker.dropped_frames} | Độ trễ TB: {worker.npu_latency_ms:.1f} ms ({worker.npu_fps:.1f} FPS)"
    )
    worker.export_csv_log()


if __name__ == "__main__":
    main()