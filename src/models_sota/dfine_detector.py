"""
Module 1 & Module 2: D-FINE (Fine-grained Distribution Refinement DETR - ICLR 2025)
Paper: Peng et al., "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement", ICLR 2025.

Đã tối ưu:
1. Contrastive Denoising Training (CDN) với Attention Mask cách ly nhóm.
2. Multi-scale Backbone (ResNet-50 hoặc MobileNetV3-Large) + Hybrid Encoder (AIFI + CCFF).
3. Fine-grained Distribution Refinement (FDR) Head tinh chỉnh tọa độ qua phân phối rời rạc (FP32 safe).
4. Suy luận End-to-End NMS-Free & tương thích 100% checkpoint đã huấn luyện.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

try:
    from torchvision.models import (
        resnet50,
        ResNet50_Weights,
        mobilenet_v3_large,
        MobileNet_V3_Large_Weights,
    )
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou_pure(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Tính ma trận Generalized IoU (GIoU) ở FP32 giữa boxes1 (N, 4) và boxes2 (M, 4) định dạng xyxy."""
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]

    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    assert dim % 4 == 0, "Hidden dim must be divisible by 4"
    grid_y = torch.arange(h, dtype=torch.float32, device=device)
    grid_x = torch.arange(w, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32, device=device) / quarter
    omega = 1.0 / (10000.0 ** omega)

    out_y = gy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = gx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pos.unsqueeze(0)


class RepConvBlock(nn.Module):
    """Khối tích chập dư chuẩn hóa dùng trong CCFF (Cross-scale Feature Fusion)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = out_ch // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class MultiScaleBackbone(nn.Module):
    """Trích xuất 3 tầng đặc trưng đa tỷ lệ (stride 8, 16, 32)."""
    def __init__(self, backbone_type: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type
        if HAS_TORCHVISION and backbone_type == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.out_channels = [512, 1024, 2048]
            self.use_tv = True
        elif HAS_TORCHVISION and backbone_type == "mobilenet_v3":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_large(weights=weights).features
            self.stage1 = net[:7]
            self.stage2 = net[7:13]
            self.stage3 = net[13:]
            self.out_channels = [40, 112, 960]
            self.use_tv = True
        else:
            self.use_tv = False
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            )
            self.c3_block = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.SiLU(inplace=True),
                RepConvBlock(256, 256),
            )
            self.c4_block = nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.SiLU(inplace=True),
                RepConvBlock(512, 512),
            )
            self.c5_block = nn.Sequential(
                nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.SiLU(inplace=True),
                RepConvBlock(1024, 1024),
            )
            self.out_channels = [256, 512, 1024]

    def forward(self, x: torch.Tensor):
        if self.use_tv and self.backbone_type == "resnet50":
            x = self.stem(x)
            x = self.layer1(x)
            c3 = self.layer2(x)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        elif self.use_tv and self.backbone_type == "mobilenet_v3":
            c3 = self.stage1(x)
            c4 = self.stage2(c3)
            c5 = self.stage3(c4)
        else:
            x = self.stem(x)
            c3 = self.c3_block(x)
            c4 = self.c4_block(c3)
            c5 = self.c5_block(c4)
        return [c3, c4, c5]


class HybridEncoder(nn.Module):
    """Hybrid Encoder của D-FINE: AIFI + CCFF."""
    def __init__(self, in_channels: list, hidden_dim: int = 256, nhead: int = 8):
        super().__init__()
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for ch in in_channels
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Đặt enable_nested_tensor=False để loại bỏ UserWarning của PyTorch khi norm_first=True
        self.aifi = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)

        self.td_p4 = RepConvBlock(hidden_dim * 2, hidden_dim)
        self.td_p3 = RepConvBlock(hidden_dim * 2, hidden_dim)

        self.down_p3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.bu_p4 = RepConvBlock(hidden_dim * 2, hidden_dim)

        self.down_p4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.bu_p5 = RepConvBlock(hidden_dim * 2, hidden_dim)

    def forward(self, feats: list):
        c3, c4, c5 = [proj(f) for proj, f in zip(self.input_proj, feats)]

        b, d, h5, w5 = c5.shape
        pos5 = build_2d_sincos_pos_embed(h5, w5, d, c5.device).to(c5.dtype)
        c5_flat = c5.flatten(2).transpose(1, 2) + pos5
        f5 = self.aifi(c5_flat).transpose(1, 2).reshape(b, d, h5, w5)

        f5_up = F.interpolate(f5, size=c4.shape[2:], mode="bilinear", align_corners=False)
        p4_td = self.td_p4(torch.cat([c4, f5_up], dim=1))

        p4_up = F.interpolate(p4_td, size=c3.shape[2:], mode="bilinear", align_corners=False)
        p3 = self.td_p3(torch.cat([c3, p4_up], dim=1))

        p4 = self.bu_p4(torch.cat([p4_td, self.down_p3(p3)], dim=1))
        p5 = self.bu_p5(torch.cat([f5, self.down_p4(p4)], dim=1))
        return [p3, p4, p5]


class FDRDecoderLayer(nn.Module):
    """Một tầng Decoder với Self-Attention (hỗ trợ attn_mask cho CDN), Cross-Attention và FFN."""
    def __init__(self, hidden_dim: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        mem_pos: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + tgt2)

        tgt2, _ = self.cross_attn(query=tgt + query_pos, key=memory + mem_pos, value=memory)
        tgt = self.norm2(tgt + tgt2)

        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


class FDRBoxHead(nn.Module):
    """Fine-grained Distribution Refinement (FDR) Head (tính toán ở FP32)."""
    def __init__(self, hidden_dim: int = 256, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4 * self.num_bins),
        )
        bins = torch.linspace(-1.0, 1.0, self.num_bins)
        self.register_buffer("bin_values", bins)

    def forward(self, feat: torch.Tensor, ref_boxes_unsigmoid: torch.Tensor):
        b, q, _ = feat.shape
        dist_logits = self.mlp(feat).float().view(b, q, 4, self.num_bins)
        probs = F.softmax(dist_logits, dim=-1)
        delta = torch.sum(probs * self.bin_values.float(), dim=-1)
        refined_unsigmoid = (ref_boxes_unsigmoid.float() + delta).clamp(-15.0, 15.0)
        refined_boxes = refined_unsigmoid.sigmoid()
        return refined_boxes, refined_unsigmoid, dist_logits


class DFINEDetector(nn.Module):
    """
    Kiến trúc D-FINE hoàn chỉnh tích hợp Contrastive Denoising Training (CDN)
    giúp hội tụ cực nhanh trên tập dữ liệu ít ảnh.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "resnet50",
        hidden_dim: int = 256,
        num_queries: int = 80,
        num_decoder_layers: int = 4,
        reg_max: int = 16,
        num_dn_groups: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_dn_groups = num_dn_groups

        self.backbone = MultiScaleBackbone(backbone_type=backbone_type, pretrained=pretrained_backbone)
        self.encoder = HybridEncoder(self.backbone.out_channels, hidden_dim=hidden_dim)

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        # Embedding cho Contrastive Denoising Queries (num_classes + 1 cho token nền âm tính)
        self.dn_label_embed = nn.Embedding(num_classes + 1, hidden_dim)

        self.ref_point_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder_layers = nn.ModuleList([
            FDRDecoderLayer(hidden_dim=hidden_dim, nhead=8)
            for _ in range(num_decoder_layers)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.fdr_heads = nn.ModuleList([
            FDRBoxHead(hidden_dim=hidden_dim, reg_max=reg_max)
            for _ in range(num_decoder_layers)
        ])

        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.enc_score_head.bias, bias_value)
        for cls_head in self.cls_heads:
            nn.init.constant_(cls_head.bias, bias_value)

    def _build_spatial_priors(self, feat_maps: list):
        memories, pos_embeds, anchors = [], [], []
        for lvl, feat in enumerate(feat_maps):
            b, d, h, w = feat.shape
            memories.append(feat.flatten(2).transpose(1, 2))
            pos_embeds.append(build_2d_sincos_pos_embed(h, w, d, feat.device).to(feat.dtype).expand(b, -1, -1))

            gy = (torch.arange(h, dtype=torch.float32, device=feat.device) + 0.5) / h
            gx = (torch.arange(w, dtype=torch.float32, device=feat.device) + 0.5) / w
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            scale = 0.05 * (2 ** lvl)
            wh = torch.full_like(xx, scale)
            anchor = torch.stack([xx, yy, wh, wh], dim=-1).reshape(1, h * w, 4)
            anchors.append(anchor.expand(b, -1, -1))

        memory = torch.cat(memories, dim=1)
        mem_pos = torch.cat(pos_embeds, dim=1)
        spatial_anchors = torch.cat(anchors, dim=1).clamp(1e-4, 1.0 - 1e-4)
        return memory, mem_pos, spatial_anchors

    def _prepare_cdn(self, targets: list, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Tạo Contrastive Denoising (CDN) queries (mẫu dương nhiễu nhẹ + mẫu âm nhiễu mạnh)."""
        max_gt = max((len(t["labels"]) for t in targets), default=0)
        if max_gt == 0 or self.num_dn_groups <= 0:
            return None

        num_dn = max_gt * 2 * self.num_dn_groups
        dn_labels = torch.full((batch_size, num_dn), self.num_classes, dtype=torch.long, device=device)
        dn_boxes = torch.full((batch_size, num_dn, 4), 0.5, dtype=torch.float32, device=device)
        dn_meta = []

        for b, t in enumerate(targets):
            n_gt = len(t["labels"])
            if n_gt == 0:
                dn_meta.append({"pos_idx": [], "gt_idx": []})
                continue

            gt_labels = t["labels"].to(device)
            gt_boxes = t["boxes"].float().to(device)
            pos_indices = []
            gt_indices = []

            for g in range(self.num_dn_groups):
                base = g * (2 * max_gt)
                noise_pos = (torch.rand_like(gt_boxes) * 2.0 - 1.0) * 0.15
                pos_b = gt_boxes.clone()
                pos_b[:, :2] += noise_pos[:, :2] * gt_boxes[:, 2:]
                pos_b[:, 2:] *= (1.0 + noise_pos[:, 2:])
                pos_b = pos_b.clamp(1e-3, 1.0 - 1e-3)

                dn_labels[b, base:base + n_gt] = gt_labels
                dn_boxes[b, base:base + n_gt] = pos_b
                pos_indices.extend(range(base, base + n_gt))
                gt_indices.extend(range(n_gt))

                neg_base = base + max_gt
                sign = torch.where(torch.rand_like(gt_boxes) > 0.5, 1.0, -1.0)
                noise_neg = sign * (0.5 + torch.rand_like(gt_boxes) * 0.5)
                neg_b = gt_boxes.clone()
                neg_b[:, :2] += noise_neg[:, :2] * gt_boxes[:, 2:]
                neg_b[:, 2:] *= (1.0 + noise_neg[:, 2:] * 0.5)
                neg_b = neg_b.clamp(1e-3, 1.0 - 1e-3)

                dn_labels[b, neg_base:neg_base + n_gt] = gt_labels
                dn_boxes[b, neg_base:neg_base + n_gt] = neg_b

            dn_meta.append({
                "pos_idx": torch.tensor(pos_indices, dtype=torch.long, device=device),
                "gt_idx": torch.tensor(gt_indices, dtype=torch.long, device=device),
            })

        dn_tgt = self.dn_label_embed(dn_labels).to(dtype)
        dn_ref_unsig = torch.log(dn_boxes / (1.0 - dn_boxes)).clamp(-15.0, 15.0)

        total_q = num_dn + self.num_queries
        attn_mask = torch.zeros((total_q, total_q), dtype=torch.bool, device=device)
        attn_mask[num_dn:, :num_dn] = True
        group_size = 2 * max_gt
        for g in range(self.num_dn_groups):
            s, e = g * group_size, (g + 1) * group_size
            attn_mask[s:e, :s] = True
            attn_mask[s:e, e:num_dn] = True

        return {
            "dn_tgt": dn_tgt,
            "dn_ref_unsig": dn_ref_unsig,
            "num_dn": num_dn,
            "dn_meta": dn_meta,
            "attn_mask": attn_mask,
        }

    def forward(self, images: torch.Tensor, targets: list = None):
        feats = self.backbone(images)
        pyr_feats = self.encoder(feats)
        memory, mem_pos, spatial_anchors = self._build_spatial_priors(pyr_feats)

        enc_logits = self.enc_score_head(memory).float()
        max_scores = enc_logits.max(dim=-1)[0]
        k = min(self.num_queries, memory.shape[1])
        topk_idx = torch.topk(max_scores, k=k, dim=1)[1]

        batch_idx = torch.arange(images.size(0), device=images.device).unsqueeze(1)
        tgt = memory[batch_idx, topk_idx].detach()
        sel_anchors = spatial_anchors[batch_idx, topk_idx].float()
        sel_enc_logits = enc_logits[batch_idx, topk_idx]

        anchor_unsig = torch.log(sel_anchors / (1.0 - sel_anchors))
        ref_unsig = (anchor_unsig + self.enc_box_head(tgt).float()).clamp(-15.0, 15.0)
        enc_boxes = ref_unsig.sigmoid()

        cdn_pack = None
        if self.training and targets is not None:
            cdn_pack = self._prepare_cdn(targets, images.size(0), images.device, tgt.dtype)

        if cdn_pack is not None:
            num_dn = cdn_pack["num_dn"]
            tgt = torch.cat([cdn_pack["dn_tgt"], tgt], dim=1)
            ref_unsig = torch.cat([cdn_pack["dn_ref_unsig"], ref_unsig], dim=1)
            attn_mask = cdn_pack["attn_mask"]
        else:
            num_dn = 0
            attn_mask = None

        aux_outputs = [{"pred_logits": sel_enc_logits, "pred_boxes": enc_boxes}]
        dn_outputs = []

        for layer, cls_head, fdr_head in zip(self.decoder_layers, self.cls_heads, self.fdr_heads):
            query_pos = self.ref_point_head(ref_unsig.sigmoid().to(tgt.dtype))
            tgt = layer(tgt, query_pos, memory, mem_pos, attn_mask=attn_mask)
            pred_logits = cls_head(tgt).float()
            pred_boxes, ref_unsig, dist_logits = fdr_head(tgt, ref_unsig.detach())

            if num_dn > 0:
                dn_outputs.append({
                    "pred_logits": pred_logits[:, :num_dn],
                    "pred_boxes": pred_boxes[:, :num_dn],
                })
                aux_outputs.append({
                    "pred_logits": pred_logits[:, num_dn:],
                    "pred_boxes": pred_boxes[:, num_dn:],
                    "dist_logits": dist_logits[:, num_dn:],
                })
            else:
                aux_outputs.append({
                    "pred_logits": pred_logits,
                    "pred_boxes": pred_boxes,
                    "dist_logits": dist_logits,
                })

        out = {
            "pred_logits": aux_outputs[-1]["pred_logits"],
            "pred_boxes": aux_outputs[-1]["pred_boxes"],
            "aux_outputs": aux_outputs[:-1],
        }
        if cdn_pack is not None:
            out["dn_outputs"] = dn_outputs
            out["dn_meta"] = cdn_pack["dn_meta"]
        return out

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, score_thresh: float = 0.35, top_k: int | None = None):
        """Suy luận NMS-Free. Trả về labels 1-indexed (1..num_classes)."""
        _, _, h, w = images.shape
        out = self.forward(images)
        probs = out["pred_logits"].float().sigmoid()
        boxes_cxcywh = out["pred_boxes"].float()

        results = []
        for b in range(images.size(0)):
            scores, labels = probs[b].max(dim=-1)
            keep = scores >= score_thresh
            b_scores = scores[keep]
            b_labels = labels[keep] + 1
            b_boxes = box_cxcywh_to_xyxy(boxes_cxcywh[b][keep])

            if top_k is not None and len(b_scores) > top_k:
                order = torch.argsort(b_scores, descending=True)[:top_k]
                b_scores = b_scores[order]
                b_labels = b_labels[order]
                b_boxes = b_boxes[order]

            b_boxes[:, [0, 2]] *= w
            b_boxes[:, [1, 3]] *= h
            b_boxes[:, [0, 2]] = b_boxes[:, [0, 2]].clamp(0, w)
            b_boxes[:, [1, 3]] = b_boxes[:, [1, 3]].clamp(0, h)
            results.append({
                "boxes": b_boxes,
                "scores": b_scores,
                "labels": b_labels,
            })
        return results


class HungarianMatcher(nn.Module):
    """Khớp cặp 1-1 tối ưu toàn cục giữa Q dự đoán và M nhãn thực tế (ổn định FP32)."""
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: list):
        pred_logits = pred_logits.float()
        pred_boxes = pred_boxes.float()
        bs, _, _ = pred_logits.shape
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"].float()
            if len(tgt_ids) == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                ))
                continue

            logits_b = pred_logits[b]
            out_prob = logits_b.sigmoid()
            out_bbox = pred_boxes[b]

            alpha, gamma = 0.25, 2.0
            neg_cost = (1.0 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-logits_b))
            pos_cost = alpha * ((1.0 - out_prob) ** gamma) * (-F.logsigmoid(logits_b))
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou_pure(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64),
            ))
        return indices


class DFINECriterion(nn.Module):
    """Hàm mất mát D-FINE (luôn chạy ở FP32): Matching Loss + Contrastive Denoising (CDN) Loss."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    def _sigmoid_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0):
        inputs = inputs.float()
        targets = targets.float()
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** gamma)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
        return loss.sum() / max(num_boxes, 1.0)

    def _compute_layer_loss(self, outputs: dict, targets: list, num_boxes: float):
        pred_logits = outputs["pred_logits"].float()
        pred_boxes = outputs["pred_boxes"].float()
        indices = self.matcher(pred_logits, pred_boxes, targets)

        target_classes_onehot = torch.zeros_like(pred_logits)
        matched_pred_boxes = []
        matched_tgt_boxes = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                tgt_labels = targets[b]["labels"][tgt_idx]
                target_classes_onehot[b, src_idx, tgt_labels] = 1.0
                matched_pred_boxes.append(pred_boxes[b, src_idx])
                matched_tgt_boxes.append(targets[b]["boxes"][tgt_idx].float())

        loss_cls = self._sigmoid_focal_loss(pred_logits, target_classes_onehot, num_boxes)

        if len(matched_pred_boxes) > 0:
            src_b = torch.cat(matched_pred_boxes, dim=0)
            tgt_b = torch.cat(matched_tgt_boxes, dim=0)
            loss_l1 = F.l1_loss(src_b, tgt_b, reduction="sum") / max(num_boxes, 1.0)
            giou = torch.diag(generalized_box_iou_pure(
                box_cxcywh_to_xyxy(src_b),
                box_cxcywh_to_xyxy(tgt_b),
            ))
            loss_giou = (1.0 - giou).sum() / max(num_boxes, 1.0)
        else:
            loss_l1 = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        return 2.0 * loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou

    def _compute_cdn_loss(self, dn_outputs: list, dn_meta: list, targets: list, num_boxes: float):
        total_cdn = 0.0
        for dn_out in dn_outputs:
            pred_logits = dn_out["pred_logits"].float()
            pred_boxes = dn_out["pred_boxes"].float()
            target_onehot = torch.zeros_like(pred_logits)
            matched_pred = []
            matched_tgt = []
            num_pos = 0

            for b, meta in enumerate(dn_meta):
                pos_idx = meta["pos_idx"]
                gt_idx = meta["gt_idx"]
                if len(pos_idx) > 0:
                    tgt_labels = targets[b]["labels"][gt_idx]
                    target_onehot[b, pos_idx, tgt_labels] = 1.0
                    matched_pred.append(pred_boxes[b, pos_idx])
                    matched_tgt.append(targets[b]["boxes"][gt_idx].float())
                    num_pos += len(pos_idx)

            denom = float(max(num_pos, 1))
            loss_cls = self._sigmoid_focal_loss(pred_logits, target_onehot, denom)
            if len(matched_pred) > 0:
                src_b = torch.cat(matched_pred, dim=0)
                tgt_b = torch.cat(matched_tgt, dim=0)
                loss_l1 = F.l1_loss(src_b, tgt_b, reduction="sum") / denom
                giou = torch.diag(generalized_box_iou_pure(
                    box_cxcywh_to_xyxy(src_b),
                    box_cxcywh_to_xyxy(tgt_b),
                ))
                loss_giou = (1.0 - giou).sum() / denom
            else:
                loss_l1 = pred_boxes.sum() * 0.0
                loss_giou = pred_boxes.sum() * 0.0

            total_cdn = total_cdn + (2.0 * loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou)
        return total_cdn / max(len(dn_outputs), 1)

    def forward(self, outputs: dict, targets: list) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            num_boxes = sum(len(t["labels"]) for t in targets)
            total_loss = self._compute_layer_loss(outputs, targets, float(num_boxes))
            for aux in outputs.get("aux_outputs", []):
                total_loss = total_loss + 0.5 * self._compute_layer_loss(aux, targets, float(num_boxes))
            if "dn_outputs" in outputs and len(outputs["dn_outputs"]) > 0:
                total_loss = total_loss + self._compute_cdn_loss(
                    outputs["dn_outputs"], outputs["dn_meta"], targets, float(num_boxes)
                )
            return total_loss


def build_dfine_detector(
    num_classes: int,
    variant: str = "dfine_s",
    pretrained_backbone: bool = True,
    num_queries: int | None = None,
) -> DFINEDetector:
    """
    Khởi tạo mô hình D-FINE:
    - 'dfine_s': MobileNetV3-Large + 3 decoder layers.
    - 'dfine_l': ResNet-50 + 4 decoder layers.
    """
    if variant == "dfine_s":
        return DFINEDetector(
            num_classes=num_classes,
            backbone_type="mobilenet_v3",
            hidden_dim=192,
            num_queries=num_queries if num_queries is not None else 40,
            num_decoder_layers=3,
            reg_max=12,
            num_dn_groups=4,
            pretrained_backbone=pretrained_backbone,
        )
    elif variant == "dfine_l":
        return DFINEDetector(
            num_classes=num_classes,
            backbone_type="resnet50",
            hidden_dim=256,
            num_queries=num_queries if num_queries is not None else 100,
            num_decoder_layers=4,
            reg_max=16,
            num_dn_groups=3,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise ValueError(f"Unknown D-FINE variant: {variant}")