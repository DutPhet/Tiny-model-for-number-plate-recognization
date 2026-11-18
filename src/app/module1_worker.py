import csv
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import (
    DEFAULT_VEHICLE_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resolve_project_paths,
)

resolve_project_paths()
from models_sota import DashcamHybridTracker, SparseOpticalFlowCMC, build_dfine_detector


class Module1RealtimeWorker:
    def __init__(
        self,
        sensor,
        ckpt_path: Path,
        device: torch.device,
        out_dir: Path,
        conf_thresh: float = 0.45,
        low_thresh: float = 0.18,
        use_fp16: bool = True,
        use_cmc: bool = True,
        use_imagenet_norm: bool = True,
    ):
        self.sensor = sensor
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.conf_thresh = float(conf_thresh)
        self.low_thresh = min(float(low_thresh), self.conf_thresh * 0.65)
        self.use_fp16 = bool(use_fp16 and device.type == "cuda")
        self.use_cmc = bool(use_cmc)

        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        self.variant = ckpt.get("variant", "dfine_s")
        default_hw = (480, 800) if self.variant == "dfine_l" else (384, 640)
        self.img_size = tuple(ckpt.get("img_size", default_hw))  # (H, W)
        num_classes = int(ckpt.get("num_classes", 4))

        raw_names = ckpt.get("class_names", None)
        if raw_names and isinstance(raw_names, (list, tuple)):
            self.class_names = {i + 1: str(name).upper() for i, name in enumerate(raw_names)}
        else:
            self.class_names = dict(DEFAULT_VEHICLE_CLASSES)

        self.model = build_dfine_detector(
            num_classes=num_classes,
            variant=self.variant,
            pretrained_backbone=False,
        ).to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])

        if self.use_fp16:
            self.model.half()

        self.use_imagenet_norm = use_imagenet_norm
        mean = ckpt.get("mean", IMAGENET_MEAN)
        std = ckpt.get("std", IMAGENET_STD)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

        self.tracker = DashcamHybridTracker(
            high_thresh=self.conf_thresh,
            low_thresh=self.low_thresh,
            iou_thresh=0.22,
            stage2_iou_thresh=0.25,
            buffer_ratio=0.18,
            ocm_weight=0.15,
            max_age=30,
            max_coast_sec=0.85,
            min_hits=2,
            use_cmc=self.use_cmc,
        )

        self.state_lock = threading.Lock()
        self.active_tracks = []
        self.track_scores = {}
        self.track_trails = defaultdict(lambda: deque(maxlen=18))
        self.raw_detections = (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

        self.track_history = {}
        self.unique_counts = defaultdict(int)

        self.npu_fps = 0.0
        self.npu_latency_ms = 0.0
        self.processed_cycles = 0
        self.dropped_frames = 0
        self.last_seq = 0
        self.stopped = False

        mode_str = f"{device.type.upper()} {'FP16' if self.use_fp16 else 'FP32'}"
        print(f"🔌 Khởi động Module 1 Realtime [{mode_str}]")
        print(f"   • Checkpoint : {self.ckpt_path}")
        print(f"   • Kiến trúc  : {self.variant.upper()} | ImgSize: {self.img_size[1]}x{self.img_size[0]}")
        print(f"   • Tracker    : Hybrid BoT-SORT/OC-SORT/ByteTrack | CMC: {'ON' if self.use_cmc else 'OFF'}")

        self.thread = threading.Thread(target=self._inference_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def toggle_cmc(self) -> bool:
        with self.state_lock:
            self.use_cmc = not self.use_cmc
            self.tracker.use_cmc = self.use_cmc
            if self.use_cmc and self.tracker.cmc is None:
                self.tracker.cmc = SparseOpticalFlowCMC()
            elif not self.use_cmc and self.tracker.cmc is not None:
                self.tracker.cmc.reset()
            return self.use_cmc

    def reset_state(self):
        with self.state_lock:
            self.tracker.reset()
            self.active_tracks = []
            self.track_scores = {}
            self.track_trails.clear()
            self.track_history.clear()
            self.unique_counts.clear()

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        t = t.half() if self.use_fp16 else t.float()
        t.div_(255.0)
        if self.use_imagenet_norm:
            t.sub_(self.mean_t).div_(self.std_t)
        return t

    @staticmethod
    def _match_scores_to_tracks(tracks: list, det_boxes: np.ndarray, det_scores: np.ndarray) -> dict:
        if len(tracks) == 0 or len(det_boxes) == 0:
            return {}
        t_boxes = np.array([t[1] for t in tracks], dtype=np.float32)
        inter_x1 = np.maximum(t_boxes[:, None, 0], det_boxes[None, :, 0])
        inter_y1 = np.maximum(t_boxes[:, None, 1], det_boxes[None, :, 1])
        inter_x2 = np.minimum(t_boxes[:, None, 2], det_boxes[None, :, 2])
        inter_y2 = np.minimum(t_boxes[:, None, 3], det_boxes[None, :, 3])
        inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        area_t = np.maximum(0.0, t_boxes[:, 2] - t_boxes[:, 0]) * np.maximum(0.0, t_boxes[:, 3] - t_boxes[:, 1])
        area_d = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0]) * np.maximum(0.0, det_boxes[:, 3] - det_boxes[:, 1])
        iou = inter / np.maximum(area_t[:, None] + area_d[None, :] - inter, 1e-6)

        score_map = {}
        for r, trk in enumerate(tracks):
            best_c = int(np.argmax(iou[r]))
            if iou[r, best_c] > 0.25:
                score_map[trk[0]] = float(det_scores[best_c])
        return score_map

    @torch.inference_mode()
    def _inference_loop(self):
        while not self.stopped:
            frame, frame_ts, seq = self.sensor.get_latest()
            if frame is None or seq == self.last_seq:
                if self.sensor.eof:
                    break
                time.sleep(0.002)
                continue

            if self.last_seq > 0 and seq > self.last_seq + 1:
                self.dropped_frames += (seq - self.last_seq - 1)
            self.last_seq = seq

            t0 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]

            inp = self._preprocess(frame)
            det = self.model.predict(inp, score_thresh=self.low_thresh)[0]

            boxes = det["boxes"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            scores = det["scores"].cpu().numpy()

            if len(boxes) > 0:
                boxes[:, [0, 2]] *= orig_w / self.img_size[1]
                boxes[:, [1, 3]] *= orig_h / self.img_size[0]

            high_mask = scores >= self.conf_thresh
            raw_high = (boxes[high_mask].copy(), labels[high_mask].copy(), scores[high_mask].copy())

            with self.state_lock:
                tracks = self.tracker.update(
                    boxes,
                    labels,
                    det_scores=scores,
                    now_ts=frame_ts,
                    frame_bgr=frame,
                )

            score_map = self._match_scores_to_tracks(tracks, boxes, scores)
            wall_now = datetime.now().strftime("%H:%M:%S")

            dt_ms = (time.perf_counter() - t0) * 1000.0
            inst_fps = 1000.0 / max(dt_ms, 1e-3)
            self.npu_latency_ms = 0.8 * self.npu_latency_ms + 0.2 * dt_ms if self.processed_cycles > 0 else dt_ms
            self.npu_fps = 0.8 * self.npu_fps + 0.2 * inst_fps if self.processed_cycles > 0 else inst_fps
            self.processed_cycles += 1

            with self.state_lock:
                self.active_tracks = tracks
                self.track_scores = score_map
                self.raw_detections = raw_high

                for tid, v_box, _, _, cls_id in tracks:
                    cls_name = self.class_names.get(cls_id, f"CLS_{cls_id}")
                    cx = int(0.5 * (v_box[0] + v_box[2]))
                    cy = int(0.5 * (v_box[1] + v_box[3]))
                    self.track_trails[tid].append((cx, cy))

                    if tid not in self.track_history:
                        self.track_history[tid] = {
                            "cls_id": cls_id,
                            "cls_name": cls_name,
                            "max_conf": score_map.get(tid, 0.0),
                            "first_seen": wall_now,
                            "last_seen": wall_now,
                            "frames": 1,
                        }
                        self.unique_counts[cls_name] += 1
                    else:
                        rec = self.track_history[tid]
                        rec["last_seen"] = wall_now
                        rec["frames"] += 1
                        if tid in score_map:
                            rec["max_conf"] = max(rec["max_conf"], score_map[tid])

    def get_snapshot(self):
        with self.state_lock:
            tracks_copy = [
                (tid, box.copy(), vel.copy(), ts, cls_id, self.track_scores.get(tid, 0.0))
                for tid, box, vel, ts, cls_id in self.active_tracks
            ]
            trails_copy = {tid: list(pts) for tid, pts in self.track_trails.items()}
            raw_copy = (
                self.raw_detections[0].copy(),
                self.raw_detections[1].copy(),
                self.raw_detections[2].copy(),
            )
            counts_copy = dict(self.unique_counts)
            total_unique = len(self.track_history)
            cmc_on = self.use_cmc
        return tracks_copy, trails_copy, raw_copy, counts_copy, total_unique, cmc_on

    def export_csv_log(self) -> Path:
        csv_path = self.out_dir / "module1_tracks.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Track_ID", "Vehicle_Class", "Max_Confidence", "Tracked_Frames", "First_Seen", "Last_Seen"])
            for tid, rec in sorted(self.track_history.items()):
                writer.writerow([
                    tid,
                    rec["cls_name"],
                    f"{rec['max_conf']:.4f}",
                    rec["frames"],
                    rec["first_seen"],
                    rec["last_seen"],
                ])
        print(f"📄 Đã xuất log kiểm tra Module 1 tại: {csv_path}")
        return csv_path

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)