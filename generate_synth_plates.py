"""
==============================================================================
Script 2: Synthetic Vietnamese License Plate Generator (Chống Rung Camera Hành Trình)
Tự động sinh dữ liệu huấn luyện LPRNet chuẩn biển số xe Việt Nam:
- Hỗ trợ biển Trắng (xe dân sự), Vàng (xe kinh doanh/vận tải), Xanh (cơ quan).
- Hỗ trợ biển dài (1 dòng) và biển vuông (2 dòng).
- Giả lập đặc thù Camera hành trình: Rung lắc (Motion blur), Méo góc (Perspective tilt),
  Nhiễu cảm biến (Sensor noise), Thay đổi độ sáng/chói đèn, Nén JPEG.
==============================================================================
"""

import os
import random
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. BẢNG KÝ TỰ BIỂN SỐ XE VIỆT NAM (CHARSET)
# ==============================================================================
# Loại bỏ J, O, Q, R, W theo quy chuẩn đăng ký biển số Việt Nam
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
    'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'
]

# Các mã tỉnh thành phổ biến tại Việt Nam
PROVINCE_CODES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "43", "47", "48", "49",
    "50", "51", "52", "53", "54", "59", "60", "61", "62", "63", "64", "65",
    "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77",
    "78", "79", "81", "82", "83", "84", "85", "86", "88", "89", "90", "92",
    "93", "94", "95", "97", "98", "99"
]

# Các ký hiệu xe quân đội (Biển đỏ): 2 chữ cái viết tắt của đơn vị quân đội
MILITARY_PREFIXES = [
    "TM", "TC", "TH", "TK", "TT", "BP", "PK", "HQ", "QK",
    "QA", "QB", "QC", "QD", "KA", "KB", "KC", "KD", "VT",
    "BB", "BC", "BK", "BL", "BT", "BV", "HA", "HB", "HC", "HD"
]

SERIES_LETTERS = [c for c in CHARS if c.isalpha()]


def get_available_font(size):
    """Tìm font chữ in đậm phù hợp nhất trên hệ thống."""
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ==============================================================================
# 2. BỘ SINH CHUỖI BIỂN SỐ NGẪU NHIÊN
# ==============================================================================
def random_plate_number():
    """
    Sinh ngẫu nhiên chuỗi biển số hợp lệ:
    - Ô tô dân sự/kinh doanh/công: 30A12345, 51K99999, 43C56789
    - Xe máy dân sự: 29G112345, 59X298765
    - Xe quân đội (Biển đỏ): TM1234, QK56789, BP1234, VT9999
    """
    # 5% tỷ lệ là biển quân đội (Biển đỏ)
    is_military = random.random() < 0.05
    if is_military:
        plate_type = "red"
        prov = ""
        seri_str = random.choice(MILITARY_PREFIXES)
        # Biển quân đội thường có 4 hoặc 5 số
        num_digits = random.choice([4, 5])
        digits = "".join([str(random.randint(0, 9)) for _ in range(num_digits)])
        is_bike = False
        raw_text = f"{seri_str}{digits}"
        return raw_text, prov, seri_str, digits, is_bike, plate_type

    # 95% là các loại biển dân sự / cơ quan / vận tải
    prov = random.choice(PROVINCE_CODES)
    seri = random.choice(SERIES_LETTERS)
    
    # 25% tỷ lệ là biển xe máy (có thêm số phụ, ví dụ G1, F2), 75% là biển ô tô
    is_bike = random.random() < 0.25
    if is_bike:
        sub_num = str(random.randint(1, 9))
        seri_str = f"{seri}{sub_num}"
    else:
        seri_str = seri

    # Dãy 5 chữ số
    digits = "".join([str(random.randint(0, 9)) for _ in range(5)])
    
    # Phân loại màu: 82% trắng, 14% vàng, 4% xanh
    rnd_color = random.random()
    if rnd_color < 0.82:
        plate_type = "white"
    elif rnd_color < 0.96:
        plate_type = "yellow"
    else:
        plate_type = "blue"

    raw_text = f"{prov}{seri_str}{digits}"
    return raw_text, prov, seri_str, digits, is_bike, plate_type


# ==============================================================================
# 3. VẼ BIỂN SỐ CƠ BẢN (BASE PLATE RENDERING)
# ==============================================================================
def render_base_plate(raw_text, prov, seri_str, digits, is_bike, plate_type):
    """
    Vẽ ảnh biển số chuẩn tỷ lệ cao trước khi áp dụng hiệu ứng và resize:
    - Biển trắng (xe cá nhân/doanh nghiệp): Nền trắng, chữ đen, viền đen.
    - Biển vàng (xe kinh doanh/vận tải/taxi): Nền vàng, chữ đen, viền đen.
    - Biển xanh (xe cơ quan nhà nước): Nền xanh dương, chữ trắng, viền trắng.
    - Biển đỏ (xe quân đội): Nền đỏ thắm, chữ trắng, viền trắng.
    """
    if plate_type == "white":
        bg_color = (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))
        text_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
        border_color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
    elif plate_type == "yellow":
        # Biển vàng
        bg_color = (random.randint(225, 255), random.randint(185, 220), random.randint(20, 50))
        text_color = (15, 15, 15)
        border_color = (25, 25, 25)
    elif plate_type == "blue":
        # Biển xanh
        bg_color = (random.randint(15, 45), random.randint(85, 135), random.randint(185, 235))
        text_color = (245, 245, 245)
        border_color = (230, 230, 230)
    else:
        # Biển đỏ (Quân đội)
        bg_color = (random.randint(175, 220), random.randint(15, 40), random.randint(20, 45))
        text_color = (245, 255, 255)
        border_color = (235, 245, 245)

    # Chiều dài và chiều cao ảnh gốc mô phỏng
    canvas_w = 470
    canvas_h = 100
    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Vẽ viền bo tròn
    border_margin = 6
    draw.rounded_rectangle(
        [(border_margin, border_margin), (canvas_w - border_margin, canvas_h - border_margin)],
        radius=8,
        outline=border_color,
        width=3
    )

    # Định dạng hiển thị chuỗi text
    font = get_available_font(52)
    if plate_type == "red":
        # Xe quân đội: ví dụ TM - 12.34 hoặc QK - 123.45
        if len(digits) == 4:
            display_text = f"{seri_str} - {digits[:2]}.{digits[2:]}"
        else:
            display_text = f"{seri_str} - {digits[:3]}.{digits[3:]}"
    else:
        # Xe dân sự: ví dụ 30A - 123.45
        display_text = f"{prov}{seri_str} - {digits[:3]}.{digits[3:]}"
    
    # Tính toán căn giữa chữ
    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    tx = (canvas_w - tw) // 2
    ty = (canvas_h - th) // 2 - 4
    draw.text((tx, ty), display_text, fill=text_color, font=font)

    # Điểm tròn mô phỏng chấm đinh tán biển số
    rivet_color = (200, 200, 200) if plate_type in ["blue", "red"] else (120, 120, 120)
    draw.ellipse([tx + (tw // 2) - 4, ty + 2, tx + (tw // 2) + 4, ty + 10], fill=rivet_color)

    return np.array(img)
    
    tx = (canvas_w - tw) // 2
    ty = (canvas_h - th) // 2 - 4
    draw.text((tx, ty), display_text, fill=text_color, font=font)

    # Điểm tròn mô phỏng chấm đinh tán biển số
    rivet_color = (120, 120, 120)
    draw.ellipse([tx + (tw // 2) - 4, ty + 2, tx + (tw // 2) + 4, ty + 10], fill=rivet_color)

    return np.array(img)


# ==============================================================================
# 4. BỘ HIỆU ỨNG GIẢ LẬP CAMERA HÀNH TRÌNH (DASHCAM DEGRADATION)
# ==============================================================================
def apply_motion_blur(image, max_kernel=7):
    """Mô phỏng rung lắc xe chạy và rung chấn camera theo trục ngang/chéo."""
    if random.random() > 0.65:
        return image
    k_size = random.choice([3, 5, max_kernel])
    kernel = np.zeros((k_size, k_size))
    # Tạo vệt mờ theo góc ngẫu nhiên (-20 đến 20 độ)
    angle = random.uniform(-20, 20)
    center = (k_size - 1) / 2
    for i in range(k_size):
        x = int(round(center + (i - center) * np.cos(np.radians(angle))))
        y = int(round(center + (i - center) * np.sin(np.radians(angle))))
        if 0 <= x < k_size and 0 <= y < k_size:
            kernel[y, x] = 1.0
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
        return cv2.filter2D(image, -1, kernel)
    return image


def apply_perspective_tilt(image, max_tilt=12):
    """Mô phỏng góc nghiêng phối cảnh khi camera gắn lệch hoặc xe rẽ cua."""
    if random.random() > 0.70:
        return image
    h, w = image.shape[:2]
    dx = random.uniform(0, max_tilt)
    dy = random.uniform(0, max_tilt * 0.4)
    
    # 4 góc nguồn và 4 góc đích
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [random.uniform(0, dx * 0.7), h - random.uniform(0, dy)],
        [w - random.uniform(0, dx * 0.7), h - random.uniform(0, dy)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_lighting_and_noise(image):
    """Mô phỏng chói đèn, tối xe, bóng râm và nhiễu hạt cảm biến."""
    img_float = image.astype(np.float32)

    # 1. Thay đổi độ sáng và tương phản ngẫu nhiên
    alpha = random.uniform(0.75, 1.25) # Contrast
    beta = random.uniform(-30, 30)     # Brightness
    img_float = np.clip(img_float * alpha + beta, 0, 255)

    # 2. Nhiễu hạt cảm biến (Sensor noise)
    if random.random() < 0.5:
        noise_sigma = random.uniform(3, 12)
        gauss = np.random.normal(0, noise_sigma, img_float.shape)
        img_float = np.clip(img_float + gauss, 0, 255)

    # 3. Nén JPEG (Nén video camera hành trình)
    img_uint8 = img_float.astype(np.uint8)
    if random.random() < 0.5:
        quality = random.randint(40, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
        img_uint8 = cv2.imdecode(encimg, 1)

    return img_uint8


# ==============================================================================
# 5. HÀM CHÍNH TẠO BỘ DỮ LIỆU OCR TOÀN DIỆN
# ==============================================================================
def generate_dataset(
    output_dir="data_source/dataset_ocr",
    n_train=10000,
    n_val=1500,
    target_size=(94, 24)
):
    """
    Sinh tập dataset hoàn chỉnh cho LPRNet:
    - output_dir/dict/chars.txt
    - output_dir/train/ (chứa ảnh .jpg chuẩn 94x24)
    - output_dir/val/
    - output_dir/train.txt (ánh xạ file -> text)
    - output_dir/val.txt
    """
    base_dir = Path(output_dir)
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    dict_dir = base_dir / "dict"

    for d in [train_dir, val_dir, dict_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Lưu bảng ký tự (Dictionary)
    chars_file = dict_dir / "chars.txt"
    with open(chars_file, "w", encoding="utf-8") as f:
        f.write("".join(CHARS))
    print(f"✓ Đã tạo từ điển ký tự ({len(CHARS)} ký tự): {chars_file}")

    tw, th = target_size

    # 2. Tiến trình sinh dữ liệu
    splits = [
        ("train", n_train, train_dir, base_dir / "train.txt"),
        ("val", n_val, val_dir, base_dir / "val.txt")
    ]

    for split_name, count, img_dir, txt_file in splits:
        print(f"\n🚀 Đang tạo tập {split_name.upper()} ({count:,} ảnh)...")
        lines = []

        for idx in range(count):
            raw_text, prov, seri_str, digits, is_bike, plate_type = random_plate_number()
            
            # Vẽ biển cơ bản (Trắng, Vàng, Xanh, Đỏ quân đội)
            img_bgr = render_base_plate(raw_text, prov, seri_str, digits, is_bike, plate_type)

            # Áp dụng các hiệu ứng camera hành trình
            img_bgr = apply_perspective_tilt(img_bgr)
            img_bgr = apply_motion_blur(img_bgr)
            img_bgr = apply_lighting_and_noise(img_bgr)

            # Resize chuẩn kích thước đầu vào LPRNet (94x24)
            plate_resized = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)

            # Lưu ảnh
            filename = f"{split_name}_{idx:06d}.jpg"
            file_path = img_dir / filename
            cv2.imwrite(str(file_path), plate_resized)

            # Lưu dòng nhãn: <tên_file> <text_nhãn>
            lines.append(f"{filename} {raw_text}\n")

            if (idx + 1) % 2500 == 0 or idx + 1 == count:
                print(f"   -> Đã hoàn thành {idx + 1:,} / {count:,} ảnh")

        with open(txt_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"✓ Đã lưu danh sách nhãn: {txt_file}")

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT SINH DỮ LIỆU HUẤN LUYỆN OCR CHO LPRNET!")
    print(f"📁 Thư mục lưu:     {base_dir.resolve()}")
    print(f"📊 Kích thước ảnh:  {tw} x {th} px (Chuẩn LPRNet)")
    print(f"📈 Tập Train:       {n_train:,} ảnh")
    print(f"📉 Tập Val:         {n_val:,} ảnh")
    print(f"🔤 Từ điển chars:   {''.join(CHARS)}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu biển số xe VN chống rung cho LPRNet")
    parser.add_argument("--output_dir", type=str, default="data_source/dataset_ocr", help="Thư mục lưu dataset")
    parser.add_argument("--train_samples", type=int, default=10000, help="Số lượng ảnh tập train")
    parser.add_argument("--val_samples", type=int, default=1500, help="Số lượng ảnh tập val")
    parser.add_argument("--width", type=int, default=94, help="Chiều rộng ảnh đầu vào LPRNet")
    parser.add_argument("--height", type=int, default=24, help="Chiều cao ảnh đầu vào LPRNet")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        n_train=args.train_samples,
        n_val=args.val_samples,
        target_size=(args.width, args.height)
    )
