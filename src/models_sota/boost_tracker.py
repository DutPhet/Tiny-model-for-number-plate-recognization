"""
Module 1 Tracking Engine: Hybrid BoT-SORT + OC-SORT + ByteTrack + CMC (Camera Motion Compensation)
Tối ưu chuyên biệt cho Edge Dashcam ALPR (1 lớp `vehicle` + Ưu tiên Top-K xe ở gần)
"""

from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_plate_sharpness(crop_bgr: np.ndarray) -> float:
    """Đo độ sắc nét của vùng biển số bằng phương sai Laplacian chuẩn hóa kích thước."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 16:
        return 0.0
    norm_w = max(16, int(w * (48.0 / h)))
    gray_norm = cv2.resize(gray, (norm_w, 48), interpolation=cv2.INTER_LINEAR)
    return float(cv2.Laplacian(gray_norm, cv2.CV_64F).var())


class SparseOpticalFlowCMC:
    """Bộ bù chuyển động camera (CMC) dùng Sparse Lucas-Kanade Optical Flow + RANSAC."""
    def __init__(self, downscale_width: int = 480, max_corners: int = 80):
        self.downscale_width = downscale_width
        self.max_corners = max_corners
        self.prev_gray = None
        self.prev_pts = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def _build_static_bg_mask(self, h: int, w: int, scale: float, exclude_boxes: np.ndarray | None) -> np.ndarray:
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[int(h * 0.86):, :] = 0  # Che nắp capo & watermark dashcam
        if exclude_boxes is not None and len(exclude_boxes) > 0:
            for box in exclude_boxes:
                x1, y1, x2, y2 = box[:4] * scale
                bw, bh = x2 - x1, y2 - y1
                ix1 = max(0, int(x1 - 0.08 * bw))
                iy1 = max(0, int(y1 - 0.08 * bh))
                ix2 = min(w, int(x2 + 0.08 * bw))
                iy2 = min(h, int(y2 + 0.08 * bh))
                if ix2 > ix1 and iy2 > iy1:
                    mask[iy1:iy2, ix1:ix2] = 0
        return mask

    def apply(self, frame_bgr: np.ndarray | None, exclude_boxes: np.ndarray | None = None) -> np.ndarray:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if frame_bgr is None or frame_bgr.size == 0:
            return identity

        orig_h, orig_w = frame_bgr.shape[:2]
        scale = min(1.0, self.downscale_width / max(orig_w, 1))
        if scale < 1.0:
            small_w = max(64, int(round(orig_w * scale)))
            small_h = max(64, int(round(orig_h * scale)))
            small_bgr = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_h, small_w = orig_h, orig_w
            small_bgr = frame_bgr

        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        bg_mask = self._build_static_bg_mask(small_h, small_w, scale, exclude_boxes)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=12, blockSize=3, mask=bg_mask
            )
            return identity

        if self.prev_pts is None or len(self.prev_pts) < 12:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=12, blockSize=3, mask=bg_mask
            )
            if self.prev_pts is None or len(self.prev_pts) < 12:
                self.prev_gray = gray
                return identity

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, winSize=(15, 15), maxLevel=2
        )

        affine = identity
        if curr_pts is not None and status is not None:
            valid = status.reshape(-1) == 1
            p0 = self.prev_pts[valid]
            p1 = curr_pts[valid]
            if len(p0) >= 8:
                M, inliers = cv2.estimateAffinePartial2D(
                    p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=500, confidence=0.99
                )
                if M is not None and inliers is not None and int(inliers.sum()) >= 6:
                    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
                    zoom = float(np.sqrt(max(det, 1e-6)))
                    tx_orig = float(M[0, 2]) / scale
                    ty_orig = float(M[1, 2]) / scale
                    if 0.88 <= zoom <= 1.12 and abs(tx_orig) < orig_w * 0.25 and abs(ty_orig) < orig_h * 0.25:
                        affine = np.array([
                            [float(M[0, 0]), float(M[0, 1]), tx_orig],
                            [float(M[1, 0]), float(M[1, 1]), ty_orig],
                        ], dtype=np.float32)

        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=12, blockSize=3, mask=bg_mask
        )
        return affine


def warp_xyxy_box(box: np.ndarray, affine: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    R = affine[:, :2]
    t = affine[:, 2]
    s = float(np.sqrt(max(R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0], 1e-6)))
    new_c = R @ np.array([cx, cy], dtype=np.float32) + t
    new_w, new_h = w * s, h * s
    return np.array([
        new_c[0] - 0.5 * new_w,
        new_c[1] - 0.5 * new_h,
        new_c[0] + 0.5 * new_w,
        new_c[1] + 0.5 * new_h,
    ], dtype=np.float32)


class BoTSORTKalmanFilter:
    """Bộ lọc Kalman 8 chiều chuẩn BoT-SORT: x = [cx, cy, w, h, vx, vy, vw, vh]^T."""
    def __init__(self, init_box_xyxy: np.ndarray):
        cx = 0.5 * (init_box_xyxy[0] + init_box_xyxy[2])
        cy = 0.5 * (init_box_xyxy[1] + init_box_xyxy[3])
        w = max(float(init_box_xyxy[2] - init_box_xyxy[0]), 2.0)
        h = max(float(init_box_xyxy[3] - init_box_xyxy[1]), 2.0)

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([
            (0.05 * w) ** 2, (0.05 * h) ** 2, (0.05 * w) ** 2, (0.05 * h) ** 2,
            (0.50 * w) ** 2, (0.50 * h) ** 2, (0.25 * w) ** 2, (0.25 * h) ** 2,
        ]).astype(np.float32)

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    def apply_cmc(self, affine: np.ndarray):
        R = affine[:, :2]
        t = affine[:, 2]
        s = float(np.sqrt(max(R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0], 1e-6)))

        self.x[:2] = R @ self.x[:2] + t
        self.x[2:4] *= s
        self.x[4:6] = R @ self.x[4:6]
        self.x[6:8] *= s

        A = np.zeros((8, 8), dtype=np.float32)
        A[:2, :2] = R
        A[2, 2] = A[3, 3] = A[6, 6] = A[7, 7] = s
        A[4:6, 4:6] = R
        self.P = A @ self.P @ A.T

    def predict(self, dt: float):
        dt = float(np.clip(dt, 1e-3, 0.5))
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt

        w = max(float(self.x[2]), 4.0)
        h = max(float(self.x[3]), 4.0)
        size_damp = 0.6 + 0.4 * min(1.0, 200.0 / max(w, h))
        q_pos_x = (0.04 * w * dt * 10.0) ** 2
        q_pos_y = (0.04 * h * dt * 10.0) ** 2
        q_vel_x = (0.25 * w * dt * 10.0 * size_damp) ** 2
        q_vel_y = (0.25 * h * dt * 10.0 * size_damp) ** 2
        Q = np.diag([
            q_pos_x, q_pos_y, q_pos_x * 0.6, q_pos_y * 0.6,
            q_vel_x, q_vel_y, q_vel_x * 0.5, q_vel_y * 0.5,
        ]).astype(np.float32)

        self.x = F @ self.x
        self.x[2] = max(float(self.x[2]), 4.0)
        self.x[3] = max(float(self.x[3]), 4.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, det_box_xyxy: np.ndarray, score: float = 0.8):
        cx = 0.5 * (det_box_xyxy[0] + det_box_xyxy[2])
        cy = 0.5 * (det_box_xyxy[1] + det_box_xyxy[3])
        w = max(float(det_box_xyxy[2] - det_box_xyxy[0]), 2.0)
        h = max(float(det_box_xyxy[3] - det_box_xyxy[1]), 2.0)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        conf_scale = 1.0 / max(float(score), 0.15)
        R_meas = np.diag([
            (0.045 * w * conf_scale) ** 2,
            (0.045 * h * conf_scale) ** 2,
            (0.055 * w * conf_scale) ** 2,
            (0.055 * h * conf_scale) ** 2,
        ]).astype(np.float32)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R_meas
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.x[2] = max(float(self.x[2]), 4.0)
        self.x[3] = max(float(self.x[3]), 4.0)
        I_KH = np.eye(8, dtype=np.float32) - (K @ self.H)
        self.P = I_KH @ self.P

    def get_box_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)

    def get_box_velocity_xyxy(self) -> np.ndarray:
        vx, vy, vw, vh = self.x[4:8]
        if abs(vx) < 8.0 and abs(vy) < 8.0:
            vx = vy = vw = vh = 0.0
        return np.array([vx - 0.5 * vw, vy - 0.5 * vh, vx + 0.5 * vw, vy + 0.5 * vh], dtype=np.float32)


class _STrack:
    def __init__(self, track_id: int, box_xyxy: np.ndarray, cls_id: int, score: float, now_ts: float):
        self.track_id = track_id
        self.cls_id = int(cls_id)
        self.score = float(score)
        self.kf = BoTSORTKalmanFilter(box_xyxy)
        self.last_obs = box_xyxy.astype(np.float32).copy()
        self.obs_history = deque(maxlen=6)
        self.obs_history.append((now_ts, self._center(box_xyxy)))

        self.last_ts = now_ts
        self.last_det_ts = now_ts
        self.hits = 1
        self.age = 0
        self.is_confirmed = False

    @staticmethod
    def _center(box: np.ndarray) -> np.ndarray:
        return np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float32)

    def warp_cmc(self, affine: np.ndarray):
        self.kf.apply_cmc(affine)
        self.last_obs = warp_xyxy_box(self.last_obs, affine)
        R = affine[:, :2]
        t = affine[:, 2]
        warped_hist = deque(maxlen=6)
        for ts, pt in self.obs_history:
            warped_hist.append((ts, R @ pt + t))
        self.obs_history = warped_hist

    def predict(self, now_ts: float):
        dt = now_ts - self.last_ts
        if dt > 0:
            self.kf.predict(dt)
            self.last_ts = now_ts
        self.age += 1

    def update(self, box_xyxy: np.ndarray, score: float, now_ts: float, min_hits: int = 2):
        self.kf.update(box_xyxy, score=score)
        self.last_obs = box_xyxy.astype(np.float32).copy()
        self.obs_history.append((now_ts, self._center(box_xyxy)))
        self.score = float(score)
        self.last_ts = now_ts
        self.last_det_ts = now_ts
        self.hits += 1
        self.age = 0
        if self.hits >= min_hits:
            self.is_confirmed = True

    def get_ocm_direction(self) -> np.ndarray | None:
        if len(self.obs_history) < 2:
            return None
        _, p_old = self.obs_history[0]
        _, p_new = self.obs_history[-1]
        diff = p_new - p_old
        norm = float(np.linalg.norm(diff))
        if norm < 6.0:
            return None
        return diff / norm


class DashcamHybridTracker:
    """
    Bộ bám vết phương tiện lai BoT-SORT + OC-SORT + ByteTrack + CMC:
    - Tối ưu 1 lớp `vehicle`: dùng ràng buộc diện tích & tỷ lệ khung hình (w/h)
      thay cho khóa cứng `cls_id` để ngăn xe máy chạy cắt ngang cướp ID của ô tô.
    - Lọc bỏ xe quá nhỏ ở xa (`min_det_area_ratio`) ngay đầu vào để tiết kiệm CPU ở biên.
    """
    def __init__(
        self,
        high_thresh: float = 0.40,
        low_thresh: float = 0.15,
        iou_thresh: float = 0.22,
        stage2_iou_thresh: float = 0.25,
        buffer_ratio: float = 0.18,
        ocm_weight: float = 0.15,
        max_age: int = 30,
        max_coast_sec: float = 0.85,
        min_hits: int = 2,
        use_cmc: bool = True,
        single_class_mode: bool = True,
        max_area_ratio: float = 4.0,
        max_ar_diff: float = 1.6,
        min_det_area_ratio: float = 0.0025,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.stage2_iou_thresh = stage2_iou_thresh
        self.buffer_ratio = buffer_ratio
        self.ocm_weight = ocm_weight
        self.max_age = max_age
        self.max_coast_sec = max_coast_sec
        self.min_hits = min_hits
        self.use_cmc = use_cmc
        self.single_class_mode = single_class_mode
        self.max_area_ratio = max_area_ratio
        self.max_ar_diff = max_ar_diff
        self.min_det_area_ratio = min_det_area_ratio

        self.cmc = SparseOpticalFlowCMC() if use_cmc else None
        self.next_id = 1
        self.stracks: list[_STrack] = []

    def reset(self):
        self.next_id = 1
        self.stracks = []
        if self.cmc is not None:
            self.cmc.reset()

    def _expand_boxes(self, boxes: np.ndarray, ratio: float) -> np.ndarray:
        if len(boxes) == 0 or ratio <= 0.0:
            return boxes
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        dw, dh = w * ratio, h * ratio
        out = boxes.copy()
        out[:, 0] -= dw
        out[:, 1] -= dh
        out[:, 2] += dw
        out[:, 3] += dh
        return out

    def _iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray, expand_ratio: float = 0.0) -> np.ndarray:
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        a = self._expand_boxes(boxes_a, expand_ratio)
        b = self._expand_boxes(boxes_b, expand_ratio)
        area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

        inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    def _geometry_consistency_penalty(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)

        tw = np.maximum(track_boxes[:, 2] - track_boxes[:, 0], 1.0)
        th = np.maximum(track_boxes[:, 3] - track_boxes[:, 1], 1.0)
        dw = np.maximum(det_boxes[:, 2] - det_boxes[:, 0], 1.0)
        dh = np.maximum(det_boxes[:, 3] - det_boxes[:, 1], 1.0)

        area_t = tw * th
        area_d = dw * dh
        area_ratio = np.maximum(area_t[:, None] / area_d[None, :], area_d[None, :] / area_t[:, None])
        ar_diff = np.abs((tw / th)[:, None] - (dw / dh)[None, :])

        penalty = np.zeros_like(area_ratio, dtype=np.float32)
        penalty[area_ratio > self.max_area_ratio] -= 5.0
        soft_mask = (area_ratio > 2.5) & (area_ratio <= self.max_area_ratio)
        penalty[soft_mask] -= 0.15 * area_ratio[soft_mask]
        penalty[ar_diff > self.max_ar_diff] -= 3.0
        return penalty

    def _compute_ocm_bonus(self, tracks: list[_STrack], det_boxes: np.ndarray) -> np.ndarray:
        bonus = np.zeros((len(tracks), len(det_boxes)), dtype=np.float32)
        if len(tracks) == 0 or len(det_boxes) == 0 or self.ocm_weight <= 0.0:
            return bonus
        det_centers = np.column_stack([
            0.5 * (det_boxes[:, 0] + det_boxes[:, 2]),
            0.5 * (det_boxes[:, 1] + det_boxes[:, 3]),
        ])
        for r, trk in enumerate(tracks):
            t_dir = trk.get_ocm_direction()
            if t_dir is None:
                continue
            last_c = _STrack._center(trk.last_obs)
            vecs = det_centers - last_c[None, :]
            norms = np.linalg.norm(vecs, axis=1)
            valid = norms >= 4.0
            if np.any(valid):
                cos_sim = (vecs[valid] @ t_dir) / norms[valid]
                bonus[r, valid] = self.ocm_weight * np.clip(cos_sim, -1.0, 1.0)
        return bonus

    def update(
        self,
        det_boxes: np.ndarray,
        det_labels: np.ndarray,
        det_scores: np.ndarray | None = None,
        now_ts: float = 0.0,
        frame_bgr: np.ndarray | None = None,
        frame_wh: tuple[int, int] | None = None,
    ):
        if det_boxes is None or len(det_boxes) == 0:
            det_boxes = np.zeros((0, 4), dtype=np.float32)
            det_labels = np.zeros((0,), dtype=np.int64)
            det_scores = np.zeros((0,), dtype=np.float32)
        else:
            det_boxes = np.asarray(det_boxes, dtype=np.float32)
            det_labels = np.asarray(det_labels, dtype=np.int64)
            if det_scores is None:
                det_scores = np.full((len(det_boxes),), 0.90, dtype=np.float32)
            else:
                det_scores = np.asarray(det_scores, dtype=np.float32)

            if self.single_class_mode:
                det_labels = np.ones_like(det_labels)

            if frame_wh is None and frame_bgr is not None:
                frame_wh = (frame_bgr.shape[1], frame_bgr.shape[0])
            if frame_wh is not None and self.min_det_area_ratio > 0.0:
                frame_area = max(float(frame_wh[0] * frame_wh[1]), 1.0)
                box_areas = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0]) * np.maximum(
                    0.0, det_boxes[:, 3] - det_boxes[:, 1]
                )
                keep_near = (box_areas / frame_area) >= self.min_det_area_ratio
                det_boxes = det_boxes[keep_near]
                det_labels = det_labels[keep_near]
                det_scores = det_scores[keep_near]

        if self.cmc is not None and frame_bgr is not None:
            affine = self.cmc.apply(frame_bgr, exclude_boxes=det_boxes)
            for trk in self.stracks:
                trk.warp_cmc(affine)

        for trk in self.stracks:
            trk.predict(now_ts)

        high_mask = det_scores >= self.high_thresh
        low_mask = (det_scores >= self.low_thresh) & (~high_mask)

        high_boxes, high_labels, high_scores = det_boxes[high_mask], det_labels[high_mask], det_scores[high_mask]
        low_boxes, low_labels, low_scores = det_boxes[low_mask], det_labels[low_mask], det_scores[low_mask]

        unmatched_track_indices = list(range(len(self.stracks)))
        unmatched_high_indices = list(range(len(high_boxes)))

        # VÒNG 1: High-Confidence Association
        if len(unmatched_track_indices) > 0 and len(high_boxes) > 0:
            kf_boxes = np.array([self.stracks[i].kf.get_box_xyxy() for i in unmatched_track_indices], dtype=np.float32)
            obs_boxes = np.array([self.stracks[i].last_obs for i in unmatched_track_indices], dtype=np.float32)

            biou_kf = self._iou_matrix(kf_boxes, high_boxes, expand_ratio=self.buffer_ratio)
            biou_obs = self._iou_matrix(obs_boxes, high_boxes, expand_ratio=self.buffer_ratio * 1.15)
            sim_mat = np.maximum(biou_kf, 0.88 * biou_obs)
            sim_mat = sim_mat + self._compute_ocm_bonus([self.stracks[i] for i in unmatched_track_indices], high_boxes)
            sim_mat = sim_mat + self._geometry_consistency_penalty(kf_boxes, high_boxes)

            if not self.single_class_mode:
                for r, t_idx in enumerate(unmatched_track_indices):
                    t_cls = self.stracks[t_idx].cls_id
                    for c in range(len(high_boxes)):
                        if int(high_labels[c]) != t_cls:
                            sim_mat[r, c] = -10.0

            row_ind, col_ind = linear_sum_assignment(-sim_mat)
            matched_t, matched_h = set(), set()

            for r, c in zip(row_ind, col_ind):
                t_idx = unmatched_track_indices[r]
                cls_ok = self.single_class_mode or (int(high_labels[c]) == self.stracks[t_idx].cls_id)
                if cls_ok and sim_mat[r, c] >= self.iou_thresh:
                    self.stracks[t_idx].update(high_boxes[c], float(high_scores[c]), now_ts, min_hits=self.min_hits)
                    matched_t.add(t_idx)
                    matched_h.add(c)

            unmatched_track_indices = [i for i in unmatched_track_indices if i not in matched_t]
            unmatched_high_indices = [c for c in unmatched_high_indices if c not in matched_h]

        # VÒNG 2: Low-Confidence Association
        confirmed_unmatched = [i for i in unmatched_track_indices if self.stracks[i].is_confirmed]
        if len(confirmed_unmatched) > 0 and len(low_boxes) > 0:
            rem_kf_boxes = np.array([self.stracks[i].kf.get_box_xyxy() for i in confirmed_unmatched], dtype=np.float32)
            rem_obs_boxes = np.array([self.stracks[i].last_obs for i in confirmed_unmatched], dtype=np.float32)

            iou_kf = self._iou_matrix(rem_kf_boxes, low_boxes, expand_ratio=self.buffer_ratio * 0.8)
            iou_obs = self._iou_matrix(rem_obs_boxes, low_boxes, expand_ratio=self.buffer_ratio * 0.8)
            sim2 = np.maximum(iou_kf, 0.90 * iou_obs)
            sim2 = sim2 + self._geometry_consistency_penalty(rem_kf_boxes, low_boxes)

            if not self.single_class_mode:
                for r, t_idx in enumerate(confirmed_unmatched):
                    t_cls = self.stracks[t_idx].cls_id
                    for c in range(len(low_boxes)):
                        if int(low_labels[c]) != t_cls:
                            sim2[r, c] = -10.0

            row_ind, col_ind = linear_sum_assignment(-sim2)
            for r, c in zip(row_ind, col_ind):
                t_idx = confirmed_unmatched[r]
                cls_ok = self.single_class_mode or (int(low_labels[c]) == self.stracks[t_idx].cls_id)
                if cls_ok and sim2[r, c] >= self.stage2_iou_thresh:
                    self.stracks[t_idx].update(low_boxes[c], float(low_scores[c]), now_ts, min_hits=self.min_hits)

        # VÒNG 3: Khởi tạo track mới
        for c in unmatched_high_indices:
            new_trk = _STrack(
                track_id=self.next_id,
                box_xyxy=high_boxes[c],
                cls_id=int(high_labels[c]),
                score=float(high_scores[c]),
                now_ts=now_ts,
            )
            if self.min_hits <= 1:
                new_trk.is_confirmed = True
            self.stracks.append(new_trk)
            self.next_id += 1

        self.stracks = [
            t for t in self.stracks
            if t.age <= self.max_age and (now_ts - t.last_det_ts) <= 2.0
        ]

        visible_tracks = []
        for t in self.stracks:
            if t.is_confirmed and (now_ts - t.last_det_ts) <= self.max_coast_sec:
                visible_tracks.append((
                    t.track_id,
                    t.kf.get_box_xyxy(),
                    t.kf.get_box_velocity_xyxy(),
                    t.last_ts,
                    t.cls_id,
                ))
        return visible_tracks


def select_priority_tracks(
    tracks: list,
    frame_w: int,
    frame_h: int,
    max_k: int = 2,
    min_box_w_ratio: float = 0.08,
    min_box_h_ratio: float = 0.08,
    locked_ids: set[int] | None = None,
) -> list:
    """
    Chỉ chọn tối đa `max_k` (1 hoặc 2) xe gần nhất & chưa khóa biển số để đưa vào Module 2 & 3.
    """
    if not tracks or max_k <= 0:
        return []

    fw = max(float(frame_w), 1.0)
    fh = max(float(frame_h), 1.0)
    locked = locked_ids or set()
    scored = []

    for item in tracks:
        tid = item[0]
        if tid in locked:
            continue

        box = item[1]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        bw = max(0.0, min(fw, x2) - max(0.0, x1))
        bh = max(0.0, min(fh, y2) - max(0.0, y1))

        if (bw / fw) < min_box_w_ratio or (bh / fh) < min_box_h_ratio:
            continue

        cx = 0.5 * (x1 + x2)
        proximity_y = np.clip(y2 / fh, 0.0, 1.0)
        area_norm = np.clip((bw * bh) / (fw * fh), 0.0, 1.0)
        center_bonus = 1.0 - min(1.0, abs(cx - 0.5 * fw) / (0.5 * fw))

        priority_score = 0.50 * proximity_y + 0.35 * np.sqrt(area_norm) + 0.15 * center_bonus
        scored.append((priority_score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_k]]


def extract_plate_search_roi(
    frame_bgr: np.ndarray,
    vehicle_box: np.ndarray,
    top_trim_ratio: float = 0.25,
    pad_ratio: float = 0.08,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Cắt vùng ảnh nhỏ trên thân dưới phương tiện ở gần để đưa vào Module 2 tìm biển số.
    Trả về (ảnh_crop_BGR, (rx1, ry1, rx2, ry2) trên hệ tọa độ khung hình gốc).
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in vehicle_box[:4]]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    rx1 = max(0, int(round(x1 - pad_ratio * bw)))
    ry1 = max(0, int(round(y1 + top_trim_ratio * bh)))
    rx2 = min(w, int(round(x2 + pad_ratio * bw)))
    ry2 = min(h, int(round(y2 + pad_ratio * bh)))

    if rx2 <= rx1 or ry2 <= ry1:
        return np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0)
    return frame_bgr[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)


LITEBoostTracker = DashcamHybridTracker