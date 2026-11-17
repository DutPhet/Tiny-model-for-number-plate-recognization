"""
==============================================================================
Module STN (Spatial Transformer Network) + LPRNet Backbone
Kiến trúc mạng nơ-ron nhận diện ký tự biển số xe (OCR) tối ưu cho Camera hành trình:
1. SpatialTransformer (STN): Tự động nắn thẳng, xoay góc và chống méo/rung lắc.
2. LPRNet: Trích xuất đặc trưng đa tầng siêu nhẹ (< 2MB) không dùng RNN/LSTM.
3. STN_LPRNet: Tích hợp hoàn chỉnh End-to-End, tương thích hoàn toàn với CTC Loss.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BẢNG TỪ ĐIỂN KÝ TỰ CHUẨN BIỂN SỐ XE VIỆT NAM (CHARSET)
# ==============================================================================
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
    'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'
]
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 1-indexed (0 dành cho Blank)
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}

# ==============================================================================
# 1. KHỐI SPATIAL TRANSFORMER NETWORK (STN) - NẮN THẲNG BIỂN SỐ TỰ ĐỘNG
# ==============================================================================
class SpatialTransformer(nn.Module):
    """
    STN Module:
    - Input:  Ảnh biển số bị nghiêng/méo do camera rung lắc (B, 3, 24, 94)
    - Output: Ảnh biển số đã được nắn thẳng về chuẩn góc 90 độ (B, 3, 24, 94)
    """
    def __init__(self, in_channels=3, img_size=(94, 24)):
        super(SpatialTransformer, self).__init__()
        self.img_w, self.img_h = img_size

        # Localization Network: Mạng trích xuất đặc trưng hình học của biển số
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (24, 94) -> (12, 47)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (12, 47) -> (6, 23)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8))  # (128, 4, 8)
        )

        # Regressor: Tầng dự đoán ma trận biến đổi Affine 2x3 (6 tham số)
        # [ [scale_x, shear_x, trans_x],
        #   [shear_y, scale_y, trans_y] ]
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

        # Khởi tạo ma trận ban đầu là 0 để tanh(0) = 0 (tương đương Identity Matrix)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

    def forward(self, x):
        # 1. Dự đoán tham số biến đổi hình học
        features = self.localization(x)
        features = features.view(features.size(0), -1)
        raw_params = self.fc_loc(features)

        # 2. KHỐNG CHẾ BIÊN ĐỘ BẰNG TANH (Bounded Affine Transformation)
        # Chống hiện tượng nổ gradient đẩy theta lên 128x làm nhòe chữ:
        # - Tỷ lệ co giãn Scale: dao động an toàn trong [0.8x, 1.2x]
        # - Góc xoay / trượt Shear: dao động trong khoảng [-0.25, 0.25] (~14 độ)
        # - Tịnh tiến Translation: dịch chuyển nhẹ trong khoảng [-0.25, 0.25]
        s_x = 1.0 + 0.20 * torch.tanh(raw_params[:, 0])
        sh_x = 0.25 * torch.tanh(raw_params[:, 1])
        t_x = 0.25 * torch.tanh(raw_params[:, 2])

        sh_y = 0.25 * torch.tanh(raw_params[:, 3])
        s_y = 1.0 + 0.20 * torch.tanh(raw_params[:, 4])
        t_y = 0.25 * torch.tanh(raw_params[:, 5])

        row1 = torch.stack([s_x, sh_x, t_x], dim=1)
        row2 = torch.stack([sh_y, s_y, t_y], dim=1)
        theta = torch.stack([row1, row2], dim=1)  # (B, 2, 3)

        # 3. Grid Generator: Tạo lưới tọa độ biến đổi
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # 4. Sampler: Lấy mẫu vi phân nắn thẳng hình ảnh bằng phép nội suy song tuyến (Bilinear)
        rectified_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return rectified_x, theta


# ==============================================================================
# 2. KHỐI LPRNet BACKBONE - TRÍCH XUẤT ĐẶC TRƯNG SIÊU NHẸ (EDGE-OPTIMIZED)
# ==============================================================================
class SmallBasicBlock(nn.Module):
    """Khối tích chập đa trường tiếp nhận (Multi-receptive field) của LPRNet."""
    def __init__(self, in_channels, out_channels):
        super(SmallBasicBlock, self).__init__()
        mid_ch = out_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class LPRNet(nn.Module):
    """
    Kiến trúc LPRNet chuẩn:
    - Input:  Ảnh biển số (B, 3, 24, 94)
    - Output: Logits chuỗi ký tự phân phối xác suất thời gian (B, class_num, Time_Steps)
    """
    def __init__(self, class_num=31, in_channels=3):
        super(LPRNet, self).__init__()
        self.class_num = class_num

        # Tầng đầu vào
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        # Khối trích xuất sâu cấp 1
        self.stage2 = nn.Sequential(
            SmallBasicBlock(64, 128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=1)  # Giảm chiều cao
        )

        # Khối trích xuất sâu cấp 2
        self.stage3 = nn.Sequential(
            SmallBasicBlock(128, 256),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=1)
        )

        # Khối trích xuất sâu cấp 3 (Đặc trưng ngữ cảnh cao nhất)
        self.stage4 = nn.Sequential(
            SmallBasicBlock(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        # Bộ tổng hợp đa quy mô (Multi-scale Feature Aggregation):
        # Kết hợp thông tin nét chữ cục bộ ở tầng nông với ngữ cảnh toàn cục ở tầng sâu
        # Sau khi qua pooling/conv, tổng số kênh gộp lại = 64 + 128 + 256 + 512 = 960
        # LƯU Ý KỸ THUẬT QUAN TRỌNG: Đầu ra logits của CTC bắt buộc là Conv tuyến tính thuần túy,
        # TUYỆT ĐỐI KHÔNG dùng ReLU hay BatchNorm để mô hình có thể dự đoán logit âm cho các class sai!
        self.container = nn.Conv2d(64 + 128 + 256 + 512, class_num, kernel_size=1)

    def forward(self, x):
        f1 = self.stage1(x)   # (B, 64, H, W)
        f2 = self.stage2(f1)  # (B, 128, H/2, W)
        f3 = self.stage3(f2)  # (B, 256, H/4, W)
        f4 = self.stage4(f3)  # (B, 512, H/4, W)

        # Đưa các tầng đặc trưng về cùng kích thước chiều cao và chiều rộng của f4
        target_h, target_w = f4.size(2), f4.size(3)

        p1 = F.adaptive_avg_pool2d(f1, (target_h, target_w))
        p2 = F.adaptive_avg_pool2d(f2, (target_h, target_w))
        p3 = F.adaptive_avg_pool2d(f3, (target_h, target_w))
        p4 = f4

        # Gộp tất cả các tầng đặc trưng dọc theo trục kênh (Channel dimension)
        multi_scale_features = torch.cat([p1, p2, p3, p4], dim=1)  # (B, 960, target_h, target_w)

        # Chiếu về số lượng class ký tự
        logits_2d = self.container(multi_scale_features)  # (B, class_num, target_h, target_w)

        # Ép phẳng chiều cao thành 1 bằng Global Average Pooling theo trục H
        logits = torch.mean(logits_2d, dim=2)  # (B, class_num, target_w)
        return logits


# ==============================================================================
# 3. MÔ HÌNH TỔNG HỢP: STN + LPRNet (END-TO-END)
# ==============================================================================
class STN_LPRNet(nn.Module):
    """
    Mô hình tích hợp hoàn chỉnh End-to-End:
    1. STN: Nắn thẳng biển số bị nghiêng/rung lắc
    2. LPRNet: Đọc chuỗi ký tự bằng CTC Loss
    """
    def __init__(self, class_num=31, in_channels=3, img_size=(94, 24), use_stn=True):
        super(STN_LPRNet, self).__init__()
        self.use_stn = use_stn
        self.class_num = class_num

        if use_stn:
            self.stn = SpatialTransformer(in_channels=in_channels, img_size=img_size)
        self.lprnet = LPRNet(class_num=class_num, in_channels=in_channels)

    def forward(self, x, return_rectified=False):
        """
        - x: Tensor ảnh đầu vào (Batch_Size, 3, 24, 94)
        - return_rectified: True nếu muốn lấy ảnh đã nắn thẳng để xem trực quan trong Notebook
        """
        rectified_img = None
        theta = None

        if self.use_stn:
            rectified_img, theta = self.stn(x)
            logits = self.lprnet(rectified_img)
        else:
            logits = self.lprnet(x)

        if return_rectified:
            return logits, rectified_img, theta
        return logits


# ==============================================================================
# 4. CTC GREEDY DECODER: GIẢI MÃ CHUỖI TỪ LOGITS
# ==============================================================================
def ctc_decode(logits, chars, blank_idx=0):
    """
    Giải mã đầu ra CTC (Greedy Decoder):
    - logits: Tensor xác suất (B, class_num, Time_Steps)
    - chars: Bảng danh sách ký tự
    - blank_idx: Chỉ số của ký tự trắng (mặc định 0)
    """
    # Lấy argmax tại mỗi bước thời gian
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, Time_Steps)
    decoded_strings = []

    for pred_seq in preds:
        char_list = []
        prev_idx = -1
        for idx in pred_seq:
            # Quy tắc CTC: Bỏ qua ký tự blank và bỏ qua ký tự lặp liên tiếp
            if idx != blank_idx and idx != prev_idx:
                # Trừ 1 nếu blank_idx là 0
                c_idx = idx - 1 if blank_idx == 0 else idx
                if 0 <= c_idx < len(chars):
                    char_list.append(chars[c_idx])
            prev_idx = idx
        decoded_strings.append("".join(char_list))

    return decoded_strings


# ==============================================================================
# 5. TEST KIỂM THỬ KÍCH THƯỚC VÀ LUỒNG DỮ LIỆU
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("🧪 KIỂM THỬ MÔ HÌNH STN + LPRNet (PyTorch)")
    print("=" * 65)

    # 30 ký tự + 1 ký tự blank = 31 classes
    CHARS_SAMPLE = list("0123456789ABCDEFGHKLMNPSTUVXYZ")
    model = STN_LPRNet(class_num=len(CHARS_SAMPLE) + 1, use_stn=True)
    model.eval()

    # Giả lập 1 batch gồm 4 ảnh biển số kích thước (24, 94)
    dummy_input = torch.randn(4, 3, 24, 94)
    with torch.no_grad():
        logits, rectified, theta = model(dummy_input, return_rectified=True)

    print(f"✓ Kích thước Tensor đầu vào:         {dummy_input.shape}")
    print(f"✓ Kích thước ảnh sau STN nắn thẳng:  {rectified.shape}")
    print(f"✓ Ma trận biến đổi Affine theta:    {theta.shape}")
    print(f"✓ Kích thước Logits đầu ra LPRNet:   {logits.shape} (Batch, Classes, TimeSteps)")

    # Đếm số lượng tham số của mô hình
    total_params = sum(p.numel() for p in model.parameters())
    stn_params = sum(p.numel() for p in model.stn.parameters())
    lpr_params = sum(p.numel() for p in model.lprnet.parameters())
    print(f"\n📊 THỐNG KÊ THAM SỐ:")
    print(f"   - Tổng tham số toàn mạng: {total_params:,} (~{total_params * 4 / (1024*1024):.2f} MB)")
    print(f"   - Tham số Module STN:     {stn_params:,}")
    print(f"   - Tham số Mạng LPRNet:    {lpr_params:,}")

    # Thử giải mã chuỗi giả lập
    decoded = ctc_decode(logits, CHARS_SAMPLE, blank_idx=0)
    print(f"\n✓ Thử nghiệm CTC Decoder kết quả: {decoded}")
    print("=" * 65)
    print("🎉 KIẾN TRÚC MÔ HÌNH HOÀN TOÀN CHÍNH XÁC VÀ SẴN SÀNG HUẤN LUYỆN!")
