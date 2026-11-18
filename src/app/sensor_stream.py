import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _resolve_video_source(source: str):
    if source.isdigit():
        return int(source), True
    if source.lower().startswith(("rtsp://", "http://", "https://")):
        return source, True

    p = Path(source)
    if p.exists():
        return str(p), False

    # Tự động tìm dự phòng trong demo_vids/ hoặc thư mục cha nếu truyền thiếu thư mục con
    fname = p.name
    fallbacks = [
        APP_DIR / fname,
        APP_DIR / "demo_vids" / fname,
        APP_DIR.parent / fname,
        APP_DIR.parent / "demo_vids" / fname,
        APP_DIR.parents[1] / fname if len(APP_DIR.parents) > 1 else APP_DIR.parent / fname,
    ]
    for fb in fallbacks:
        if fb.is_file():
            print(f"ℹ️ Tự động tìm thấy video tại: {fb}")
            return str(fb), False

    raise FileNotFoundError(f"Không tìm thấy file video nguồn: {source} (đường dẫn đầy đủ: {p.resolve()})")


class DashcamSensorStream:
    def __init__(self, source: str, loop_video: bool = False, speed: float = 1.0):
        self.source = source
        self.cap_src, self.is_live = _resolve_video_source(source)
        self.loop_video = loop_video
        self.speed = max(0.1, float(speed))

        self.cap = cv2.VideoCapture(self.cap_src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở nguồn camera/video: {self.cap_src}")

        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if (raw_fps and 1.0 < raw_fps <= 120.0 and not np.isnan(raw_fps)) else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.frame_seq = 0
        self.sensor_fps = self.fps
        self.stopped = False
        self.eof = False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Không đọc được khung hình đầu tiên từ: {self.cap_src}")
        self.latest_frame = frame
        self.latest_ts = time.perf_counter()
        self.frame_seq = 1

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture_loop(self):
        frame_interval = 1.0 / (self.fps * self.speed)
        next_tick = time.perf_counter() + frame_interval
        last_stat_ts = time.perf_counter()
        stat_frames = 0

        while not self.stopped:
            if not self.is_live:
                now = time.perf_counter()
                sleep_dt = next_tick - now
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                next_tick += frame_interval

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.is_live and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_tick = time.perf_counter() + frame_interval
                    continue
                self.eof = True
                break

            now_ts = time.perf_counter()
            with self.lock:
                self.latest_frame = frame
                self.latest_ts = now_ts
                self.frame_seq += 1

            stat_frames += 1
            if now_ts - last_stat_ts >= 0.5:
                self.sensor_fps = stat_frames / (now_ts - last_stat_ts)
                stat_frames = 0
                last_stat_ts = now_ts

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, 0
            return self.latest_frame.copy(), self.latest_ts, self.frame_seq

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.cap.release()