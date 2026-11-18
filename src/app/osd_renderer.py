import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import CLASS_COLORS, DEFAULT_VEHICLE_CLASSES


def render_module1_osd(
    frame_bgr: np.ndarray,
    frame_ts: float,
    sensor,
    worker,
    tracks: list,
    trails: dict,
    raw_detections: tuple,
    unique_counts: dict,
    total_unique: int,
    cmc_on: bool,
    show_raw_only: bool = False,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    class_names = getattr(worker, "class_names", DEFAULT_VEHICLE_CLASSES)

    active_by_class = {"CAR": 0, "MOTO": 0, "BUS": 0, "TRUCK": 0}

    if show_raw_only:
        raw_boxes, raw_labels, raw_scores = raw_detections
        for box, cls_id, conf in zip(raw_boxes, raw_labels, raw_scores):
            cls_id = int(cls_id)
            cls_name = class_names.get(cls_id, f"CLS_{cls_id}")
            if cls_name in active_by_class:
                active_by_class[cls_name] += 1

            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tag = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_top = max(0, y1 - th - 10)
            cv2.rectangle(vis, (x1, y_top), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis, tag, (x1 + 5, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)
    else:
        for tid, box, vel, last_ts, cls_id, conf in tracks:
            cls_name = class_names.get(cls_id, f"CLS_{cls_id}")
            if cls_name in active_by_class:
                active_by_class[cls_name] += 1

            dt = float(np.clip(frame_ts - last_ts, 0.0, 0.25))
            smooth_box = box + vel * dt
            x1 = max(0, int(smooth_box[0]))
            y1 = max(0, int(smooth_box[1]))
            x2 = min(w, int(smooth_box[2]))
            y2 = min(h, int(smooth_box[3]))
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            pts = trails.get(tid, [])
            if len(pts) >= 2:
                for i in range(1, len(pts)):
                    cv2.line(vis, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            conf_str = f" {conf:.2f}" if conf > 0.0 else ""
            tag = f"#{tid} {cls_name}{conf_str}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_top = max(0, y1 - th - 10)
            cv2.rectangle(vis, (x1, y_top), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis, tag, (x1 + 5, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 12, 16), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    if int(time.perf_counter() * 2) % 2 == 0:
        cv2.circle(vis, (22, 21), 7, (0, 0, 255), -1)
    cv2.putText(vis, "M1 TEST", (36, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    mode_label = "RAW DETECTOR" if show_raw_only else f"TRACKER + CMC:{'ON' if cmc_on else 'OFF'}"
    status_str = (
        f"{datetime.now().strftime('%H:%M:%S')} | CAM: {sensor.sensor_fps:.0f} FPS | "
        f"AI: {worker.npu_fps:.1f} FPS ({worker.npu_latency_ms:.1f}ms) | {mode_label}"
    )
    cv2.putText(vis, status_str, (128, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 190), 2, cv2.LINE_AA)

    panel_w, panel_h = 285, 148
    x0, y0 = 12, h - panel_h - 12
    if y0 > 50:
        ov2 = vis.copy()
        cv2.rectangle(ov2, (x0, y0), (x0 + panel_w, y0 + panel_h), (12, 12, 16), -1)
        cv2.addWeighted(ov2, 0.74, vis, 0.26, 0, vis)

        cv2.putText(
            vis, f"MODULE 1 VEHICLE ({worker.variant.upper()})",
            (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA
        )

        for idx, (cname, cid) in enumerate([("CAR", 1), ("MOTO", 2), ("BUS", 3), ("TRUCK", 4)]):
            ry = y0 + 46 + idx * 21
            c_bgr = CLASS_COLORS.get(cid, (255, 255, 255))
            cv2.circle(vis, (x0 + 16, ry - 4), 5, c_bgr, -1)
            line = f"{cname:<6}: Now {active_by_class[cname]:>2} | Total {unique_counts.get(cname, 0):>3}"
            cv2.putText(vis, line, (x0 + 28, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.putText(
            vis, f"Unique Tracks: {total_unique}  [T:Raw | C:CMC | Q:Quit]",
            (x0 + 10, y0 + 136), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA
        )

    return vis