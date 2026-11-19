"""
Module 3: SVTRv2 (Scene Text Recognition Vision Transformer v2 - ICCV 2025)
Paper: Du et al., "SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition", ICCV 2025.

Đã tối ưu & sửa lỗi:
1. Đồng bộ đầu ra hàm trải phẳng biển 2 dòng `unwrap_two_line_plate` về `(64, 256)` (T=64 bước CTC).
2. Cập nhật `DEFAULT_VN_CHARS` (bổ sung 'R' cho biển rơ-moóc) và `VN_PLATE_REGEX` hỗ trợ đầy đủ biển ô tô, xe máy (VD: 29B112345, 29MD112345), biển ngoại giao (80123NG01), biển quân đội (TM1234).
3. Sửa điều kiện Khóa Sớm (`Early Lock`) dựa trên độ tin cậy trung bình tích lũy và ứng viên dẫn đầu theo `track_id`.
4. Bổ sung `preprocess_plate_ocr()` chuẩn hóa đầu vào suy luận.
"""

import re
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bảng ký tự chuẩn cho biển số Việt Nam (index 0 dành riêng cho CTC blank)
DEFAULT_VN_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
DEFAULT_IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(DEFAULT_VN_CHARS)}
DEFAULT_CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(DEFAULT_VN_CHARS)}

# Bao phủ đầy đủ các định dạng biển số Việt Nam (sau khi loại bỏ ký tự phân cách '-' và '.')
VN_PLATE_REGEX = re.compile(
    r"^(?:"
    r"[0-9]{2}[A-Z]{1,2}[0-9]{4,5}|"             # Ô tô & xe máy 2 chữ (VD: 30A12345, 29AB12345)
    r"[0-9]{2}[A-Z]{1,2}[0-9][0-9]{4,5}|"        # Xe máy sê-ri 1-2 chữ + 1 số (VD: 29B112345, 29MD112345)
    r"[0-9]{2}[0-9]{3}(?:NG|QT|NN|CV)[0-9]{2}|"  # Ngoại giao / Quốc tế / Nước ngoài / Công vụ (VD: 80123NG01)
    r"[A-Z]{2}[0-9]{4,5}"                        # Quân đội (VD: TM1234, KA12345)
    r")$"
)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMixingBlock(nn.Module):
    """Khối trộn đặc trưng cục bộ (Depthwise Conv + Inverted Bottleneck MLP)."""
    def __init__(self, dim: int, mlp_ratio: float = 2.5, drop: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = ConvBNAct(dim, dim, k=3, s=1, p=1, groups=dim)
        self.pwconv1 = ConvBNAct(dim, hidden, k=1, s=1, p=0)
        self.drop = nn.Dropout(drop)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.pwconv2(self.drop(self.pwconv1(x)))
        return shortcut + x


class GlobalMixingBlock(nn.Module):
    """Khối trộn ngữ cảnh toàn cục dùng Multi-Head Self-Attention."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.5, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        t_norm = self.norm1(tokens)
        attn_out, _ = self.attn(t_norm, t_norm, t_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class SVTRv2OCR(nn.Module):
    """
    Kiến trúc SVTRv2-VN tối ưu cho đọc chuỗi biển số Việt Nam 1 dòng & 2 dòng:
    - Input : (B, 3, 64, 256)
    - Output: (T=64, B, num_classes) dùng trực tiếp với nn.CTCLoss(blank=0)
    """
    def __init__(
        self,
        num_classes: int,
        dims: tuple = (128, 256, 384),
        depths: tuple = (3, 4, 4),
        num_heads: tuple = (4, 8, 12),
        drop_rate: float = 0.15,
    ):
        super().__init__()
        d1, d2, d3 = dims

        # Progressive Overlapping Patch Embedding: (64, 256) -> (16, 64)
        self.patch_embed = nn.Sequential(
            ConvBNAct(3, d1 // 2, k=3, s=2, p=1),
            ConvBNAct(d1 // 2, d1, k=3, s=2, p=1),
        )

        # Stage 1: Local Mixing trên độ phân giải (16, 64)
        self.stage1 = nn.Sequential(*[
            LocalMixingBlock(d1, drop=drop_rate) for _ in range(depths[0])
        ])

        # Downsample chiều cao (16, 64) -> (8, 64)
        self.down1 = ConvBNAct(d1, d2, k=3, s=(2, 1), p=1)

        # Stage 2: Đan xen Local + Global Mixing trên (8, 64)
        s2_blocks = []
        for i in range(depths[1]):
            if i % 2 == 0:
                s2_blocks.append(LocalMixingBlock(d2, drop=drop_rate))
            else:
                s2_blocks.append(GlobalMixingBlock(d2, num_heads=num_heads[1], drop=drop_rate))
        self.stage2 = nn.Sequential(*s2_blocks)

        # Downsample chiều cao (8, 64) -> (4, 64)
        self.down2 = ConvBNAct(d2, d3, k=3, s=(2, 1), p=1)

        # Stage 3: Global Mixing chuyên sâu trên (4, 64)
        self.stage3 = nn.Sequential(*[
            GlobalMixingBlock(d3, num_heads=num_heads[2], drop=drop_rate) for _ in range(depths[2])
        ])

        # Multi-scale Feature Fusion & Height Collapse -> (1, 64)
        self.height_collapse = nn.Sequential(
            ConvBNAct(d1 + d2 + d3, d3, k=1, s=1, p=0),
            ConvBNAct(d3, d3, k=(4, 1), s=1, p=0),
        )

        self.head_norm = nn.LayerNorm(d3)
        self.head_drop = nn.Dropout(drop_rate)
        self.ctc_head = nn.Linear(d3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stage1(self.patch_embed(x))  # (B, d1, 16, 64)
        f2 = self.stage2(self.down1(f1))       # (B, d2, 8, 64)
        f3 = self.stage3(self.down2(f2))       # (B, d3, 4, 64)

        target_hw = (4, f3.shape[3])
        f1_pool = F.adaptive_avg_pool2d(f1, target_hw)
        f2_pool = F.adaptive_avg_pool2d(f2, target_hw)
        f3_pool = F.adaptive_avg_pool2d(f3, target_hw)

        fused = self.height_collapse(torch.cat([f1_pool, f2_pool, f3_pool], dim=1))  # (B, d3, 1, 64)
        seq = fused.squeeze(2).permute(2, 0, 1)  # (T=64, B, d3)
        logits = self.ctc_head(self.head_drop(self.head_norm(seq)))
        return F.log_softmax(logits, dim=-1)


def unwrap_two_line_plate(plate_bgr: np.ndarray, out_h: int = 64, out_w: int = 256) -> np.ndarray:
    """
    Chuyển đổi biển số 2 dòng (xe máy hoặc ô tô biển vuông) thành dải ngang 1 dòng.
    Đồng bộ mặc định về kích thước `(64, 256)` khớp với kiến trúc `SVTRv2OCR` (`T=64`).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    h, w = plate_bgr.shape[:2]
    # Biển 2 dòng vuông có tỷ lệ h/w thường >= 0.65 (biển 1 dòng thường <= 0.52)
    if (h / max(w, 1)) > 0.65 and h >= 20:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        row_energy = sobel_x.mean(axis=1)

        y_start = int(h * 0.35)
        y_end = max(y_start + 1, int(h * 0.65))
        split_y = y_start + int(np.argmin(row_energy[y_start:y_end]))

        top_crop = plate_bgr[:min(h, split_y + 2), :]
        bot_crop = plate_bgr[max(0, split_y - 2):, :]

        half_w = out_w // 2
        top_resized = cv2.resize(top_crop, (half_w, out_h), interpolation=cv2.INTER_CUBIC)
        bot_resized = cv2.resize(bot_crop, (out_w - half_w, out_h), interpolation=cv2.INTER_CUBIC)
        return np.concatenate([top_resized, bot_resized], axis=1)

    return cv2.resize(plate_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def preprocess_plate_ocr(
    plate_bgr: np.ndarray,
    img_size: tuple = (64, 256),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Tiền xử lý ảnh biển số chuẩn hóa theo ImageNet mean/std cho SVTRv2."""
    unwrapped = unwrap_two_line_plate(plate_bgr, out_h=img_size[0], out_w=img_size[1])
    rgb = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)


def ctc_greedy_decode(log_probs: torch.Tensor, idx_to_char: dict = DEFAULT_IDX_TO_CHAR) -> list:
    """Giải mã CTC Greedy kèm tính điểm tin cậy trung bình của các ký tự được chọn."""
    probs = log_probs.exp()
    max_probs, max_indices = probs.max(dim=-1)  # (T, B)
    T, B = max_indices.shape
    results = []

    for b in range(B):
        chars = []
        confs = []
        prev_idx = 0
        for t in range(T):
            idx = int(max_indices[t, b].item())
            prob = float(max_probs[t, b].item())
            if idx != 0 and idx != prev_idx:
                ch = idx_to_char.get(idx, "")
                if ch:
                    chars.append(ch)
                    confs.append(prob)
            prev_idx = idx
        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        results.append((text, avg_conf))
    return results


class PlateVoteAggregator:
    """
    Bộ tổng hợp kết quả đọc biển số theo thời gian gắn với `track_id`:
    - Loại bỏ ký tự nhiễu ngoài chuẩn [A-Z0-9].
    - Cộng dồn điểm tin cậy có trọng số theo độ nét ảnh (Laplacian sharpness) và Regex biển số Việt Nam.
    - Hỗ trợ Khóa Sớm (Early Lock) khi một biển số hợp lệ đạt đủ số phiếu với độ tin cậy trung bình cao.
    """
    def __init__(self, lock_min_votes: int = 3, lock_conf_thresh: float = 0.85, min_sharpness: float = 35.0):
        self.lock_min_votes = lock_min_votes
        self.lock_conf_thresh = lock_conf_thresh
        self.min_sharpness = min_sharpness
        self.scores = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.conf_sums = defaultdict(lambda: defaultdict(float))
        self.locked_plates = {}

    @property
    def locked_ids(self) -> set:
        return set(self.locked_plates.keys())

    def is_locked(self, track_id: int) -> bool:
        return track_id in self.locked_plates

    def update(self, track_id: int, raw_text: str, ocr_conf: float, sharpness: float) -> str | None:
        if track_id in self.locked_plates:
            return self.locked_plates[track_id]

        clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if len(clean) < 6 or sharpness < self.min_sharpness or ocr_conf < 0.45:
            return self.get_best(track_id)

        is_valid_vn = bool(VN_PLATE_REGEX.match(clean))
        sharpness_factor = min(sharpness / 100.0, 2.5)
        regex_bonus = 1.6 if is_valid_vn else 0.85
        weight = ocr_conf * sharpness_factor * regex_bonus

        self.scores[track_id][clean] += weight
        self.counts[track_id][clean] += 1
        self.conf_sums[track_id][clean] += ocr_conf

        votes = self.counts[track_id][clean]
        avg_conf = self.conf_sums[track_id][clean] / max(1, votes)
        best_now = self.get_best(track_id)

        if (
            clean == best_now
            and votes >= self.lock_min_votes
            and avg_conf >= self.lock_conf_thresh
            and is_valid_vn
        ):
            self.locked_plates[track_id] = clean

        return best_now

    def get_best(self, track_id: int) -> str | None:
        if track_id in self.locked_plates:
            return self.locked_plates[track_id]
        candidates = self.scores.get(track_id)
        if not candidates:
            return None
        return max(candidates.items(), key=lambda kv: kv[1])[0]


def build_svtrv2_ocr(num_classes: int = len(DEFAULT_VN_CHARS) + 1, variant: str = "svtrv2_base") -> SVTRv2OCR:
    variant = variant.lower().strip()
    if variant == "svtrv2_tiny":
        return SVTRv2OCR(
            num_classes=num_classes,
            dims=(96, 192, 256),
            depths=(2, 3, 3),
            num_heads=(3, 6, 8),
            drop_rate=0.1,
        )
    return SVTRv2OCR(
        num_classes=num_classes,
        dims=(128, 256, 384),
        depths=(3, 4, 4),
        num_heads=(4, 8, 12),
        drop_rate=0.15,
    )