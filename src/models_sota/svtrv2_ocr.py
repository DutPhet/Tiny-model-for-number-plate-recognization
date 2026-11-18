"""
Module 3: SVTRv2 (Scene Text Recognition Vision Transformer v2 - ICCV 2025)
Paper: Du et al., "SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition", ICCV 2025.

Đặc điểm kiến trúc:
1. Progressive Overlapping Patch Embedding: Giữ trọn nét mảnh và góc bo ký tự biển số (0/D, 5/S, 8/B).
2. Hierarchical Local-Global Mixing:
   - Giai đoạn nông dùng Local Conv Mixing trích xuất nét từng ký tự.
   - Giai đoạn sâu dùng Global Multi-Head Self-Attention nắm bắt bố cục toàn biển số.
3. Feature Rearrangement & Multi-Scale Global Context + CTC Decoder:
   - Nén chiều cao bảo toàn 64 bước thời gian ngang (T=64), giải mã song song siêu nhanh qua nn.CTCLoss.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            GlobalMixingBlock(d3, num_heads=num_heads[2], drop=drop_rate)
            for _ in range(depths[2])
        ])

        # Feature Rearrangement & Multi-Scale Fusion (Nén chiều cao về 1, giữ T=64)
        self.height_collapse = nn.Sequential(
            ConvBNAct(d1 + d2 + d3, d3, k=1, s=1, p=0),
            ConvBNAct(d3, d3, k=(4, 1), s=1, p=0),  # (4, 64) -> (1, 64)
        )
        self.ctc_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(d3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stage1(self.patch_embed(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))

        target_hw = (4, f3.shape[3])
        f1_pool = F.adaptive_avg_pool2d(f1, target_hw)
        f2_pool = F.adaptive_avg_pool2d(f2, target_hw)
        f3_pool = F.adaptive_avg_pool2d(f3, target_hw)
        fused = self.height_collapse(torch.cat([f1_pool, f2_pool, f3_pool], dim=1))

        seq = fused.squeeze(2).permute(2, 0, 1)
        logits = self.ctc_head(seq)
        return F.log_softmax(logits, dim=-1)


import re
from collections import defaultdict

VN_PLATE_REGEX = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,5}$|^[0-9]{2}(NG|QT|NN|CV)[0-9]{4,5}$|^[A-Z]{2}[0-9]{4,5}$")


def unwrap_two_line_plate(img_bgr: np.ndarray, out_h: int = 48, out_w: int = 160) -> np.ndarray:
    """Trải phẳng thông minh biển số vuông 2 dòng thành dải ngang."""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if (h / max(w, 1)) > 0.72 and h >= 20:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        row_energy = sobel_x.mean(axis=1)
        y_start = max(4, int(h * 0.38))
        y_end = min(h - 4, int(h * 0.62))
        split_y = (y_start + int(np.argmin(row_energy[y_start:y_end]))) if y_end > y_start else (h // 2)

        pad = max(1, int(h * 0.03))
        top = cv2.resize(img_bgr[:min(h, split_y + pad), :], (out_w // 2, out_h))
        bot = cv2.resize(img_bgr[max(0, split_y - pad):, :], (out_w - (out_w // 2), out_h))
        return np.hstack([top, bot])

    return cv2.resize(img_bgr, (out_w, out_h))


def ctc_greedy_decode(log_probs: torch.Tensor, idx_to_char: dict) -> list[tuple[str, float]]:
    """Giải mã chuỗi CTC trả về danh sách (chuỗi_biển_số, độ_tin_cậy_trung_bình)."""
    probs = log_probs.exp()
    max_probs, max_indices = probs.max(dim=-1)
    preds = max_indices.detach().permute(1, 0).cpu().numpy()
    confs = max_probs.detach().permute(1, 0).cpu().numpy()

    decoded_list = []
    for seq, seq_conf in zip(preds, confs):
        chars, char_confs = [], []
        prev = -1
        for idx, c_prob in zip(seq, seq_conf):
            idx_int = int(idx)
            if idx_int != 0 and idx_int != prev:
                chars.append(idx_to_char[idx_int])
                char_confs.append(float(c_prob))
            prev = idx_int
        text = "".join(chars)
        avg_conf = float(np.mean(char_confs)) if char_confs else 0.0
        decoded_list.append((text, avg_conf))
    return decoded_list

class PlateVoteAggregator:
    """
    Gom phiếu đa khung hình theo `track_id` (chỉ dành cho Top 1–2 xe gần nhất).
    Tích hợp lọc độ mờ Laplacian và Khóa Sớm (`Early Lock`) để giải phóng tài nguyên biên.
    """
    def __init__(self, min_sharpness: float = 45.0, lock_min_votes: int = 3, lock_conf_thresh: float = 0.85):
        self.min_sharpness = min_sharpness
        self.lock_min_votes = lock_min_votes
        self.lock_conf_thresh = lock_conf_thresh
        self.scores: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: dict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.locked_plates: dict[int, str] = {}

    @property
    def locked_ids(self) -> set[int]:
        return set(self.locked_plates.keys())

    def update(self, track_id: int, raw_text: str, ocr_conf: float, sharpness: float) -> str | None:
        if track_id in self.locked_plates:
            return self.locked_plates[track_id]

        clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if len(clean) < 7 or sharpness < self.min_sharpness or ocr_conf < 0.45:
            return self.get_best(track_id)

        sharp_weight = min(sharpness / 120.0, 1.5)
        regex_bonus = 1.25 if VN_PLATE_REGEX.match(clean) else 0.85
        self.scores[track_id][clean] += float(ocr_conf) * sharp_weight * regex_bonus
        self.counts[track_id][clean] += 1

        if (
            self.counts[track_id][clean] >= self.lock_min_votes
            and ocr_conf >= self.lock_conf_thresh
            and VN_PLATE_REGEX.match(clean)
        ):
            self.locked_plates[track_id] = clean

        return self.get_best(track_id)

    def get_best(self, track_id: int) -> str | None:
        if track_id in self.locked_plates:
            return self.locked_plates[track_id]
        if track_id not in self.scores or not self.scores[track_id]:
            return None
        return max(self.scores[track_id].items(), key=lambda kv: kv[1])[0]

    def prune(self, active_track_ids: set[int]):
        stale = [tid for tid in list(self.scores.keys()) if tid not in active_track_ids]
        for tid in stale:
            self.scores.pop(tid, None)
            self.counts.pop(tid, None)
            self.locked_plates.pop(tid, None)

def build_svtrv2_ocr(num_classes: int, variant: str = "svtrv2_base") -> SVTRv2OCR:
    """
    Khởi tạo SVTRv2:
    - 'svtrv2_base': Độ chính xác cao nhất (dims 128-256-384, ~6.4M params)
    - 'svtrv2_tiny': Siêu nhẹ chạy biên (dims 64-128-256, ~1.8M params)
    """
    if variant == "svtrv2_base":
        return SVTRv2OCR(
            num_classes=num_classes,
            dims=(128, 256, 384),
            depths=(3, 4, 4),
            num_heads=(4, 8, 12),
            drop_rate=0.15,
        )
    elif variant == "svtrv2_tiny":
        return SVTRv2OCR(
            num_classes=num_classes,
            dims=(64, 128, 256),
            depths=(2, 3, 3),
            num_heads=(2, 4, 8),
            drop_rate=0.10,
        )
    else:
        raise ValueError(f"Unknown SVTRv2 variant: {variant}")