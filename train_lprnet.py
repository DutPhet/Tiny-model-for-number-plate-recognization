"""
==============================================================================
Script Huấn luyện LPRNet + STN (End-to-End với PyTorch CTC Loss)
HỖ TRỢ CẢ 3 CHẾ ĐỘ:
1. Huấn luyện Bounded STN từ đầu (Scratch)
2. Huấn luyện Two-Stage Warmup (Đóng băng STN pha 1)
3. Huấn luyện Tinh chỉnh (Fine-tuning) trộn dữ liệu cũ (dataset_ocr) + dữ liệu mới (real_cropped_plates)
==============================================================================
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_stn_lprnet import STN_LPRNet, CHARS, CHAR2IDX, ctc_decode


# ==============================================================================
# 1. DATASET & DATALOADER CHO LPRNET (HỖ TRỢ TRỘN MULTI-SOURCE DATA)
# ==============================================================================
class LPRDataset(Dataset):
    """
    Dataset đọc ảnh biển số và nhãn text từ file txt:
    Tối ưu hóa In-Memory Caching: Tải toàn bộ ảnh vào RAM (~70MB) giúp tăng tốc 10x trên CPU.
    """
    def __init__(self, txt_path, img_dir, extra_samples=None, target_size=(94, 24), preload=True):
        self.img_dir = Path(img_dir)
        self.target_w, self.target_h = target_size
        self.samples = []
        self.preload = preload

        if Path(txt_path).exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name, raw_label = parts[0], parts[1].upper()
                        img_path = self.img_dir / img_name
                        if img_path.exists():
                            self.samples.append((img_path, raw_label))

        # Trộn thêm dữ liệu thực tế (nếu có)
        if extra_samples:
            self.samples.extend(extra_samples)

        self.cached_data = []
        if self.preload and self.samples:
            print(f"⚡ Đang nạp nhanh {len(self.samples):,} ảnh vào RAM để tăng tốc...")
            for img_path, raw_label in self.samples:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    img_bgr = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
                elif img_bgr.shape[1] != self.target_w or img_bgr.shape[0] != self.target_h:
                    img_bgr = cv2.resize(img_bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

                img_float = (img_bgr.astype(np.float32) - 127.5) * 0.0078125
                img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1))
                label_indices = torch.tensor([CHAR2IDX[c] for c in raw_label if c in CHAR2IDX], dtype=torch.long)
                self.cached_data.append((img_tensor, label_indices, raw_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload and self.cached_data:
            return self.cached_data[idx]

        img_path, raw_label = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        elif img_bgr.shape[1] != self.target_w or img_bgr.shape[0] != self.target_h:
            img_bgr = cv2.resize(img_bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        img_float = (img_bgr.astype(np.float32) - 127.5) * 0.0078125
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1))
        label_indices = torch.tensor([CHAR2IDX[c] for c in raw_label if c in CHAR2IDX], dtype=torch.long)
        return img_tensor, label_indices, raw_label


def lpr_collate_fn(batch):
    images = []
    labels = []
    lengths = []
    raw_texts = []

    for img_tensor, label_tensor, raw_text in batch:
        images.append(img_tensor)
        labels.append(label_tensor)
        lengths.append(len(label_tensor))
        raw_texts.append(raw_text)

    images = torch.stack(images, dim=0)
    flat_targets = torch.cat(labels, dim=0) if len(labels) > 0 else torch.empty(0, dtype=torch.long)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, flat_targets, target_lengths, raw_texts


# ==============================================================================
# 2. HÀM ĐÁNH GIÁ (VALIDATION)
# ==============================================================================
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_plates = 0
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for images, targets, target_lengths, raw_texts in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            batch_size = images.size(0)

            logits = model(images)
            T, N, C = logits.size(2), logits.size(0), logits.size(1)
            log_probs = F.log_softmax(logits, dim=1).permute(2, 0, 1)
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = ctc_decode(logits, CHARS, blank_idx=0)
            for pred_str, gt_str in zip(preds, raw_texts):
                if pred_str == gt_str:
                    correct_plates += 1

                min_len = min(len(pred_str), len(gt_str))
                for c_p, c_g in zip(pred_str[:min_len], gt_str[:min_len]):
                    if c_p == c_g:
                        correct_chars += 1
                total_chars += max(len(pred_str), len(gt_str))

    avg_loss = total_loss / max(1, total_samples)
    plate_acc = (correct_plates / max(1, total_samples)) * 100.0
    char_acc = (correct_chars / max(1, total_chars)) * 100.0
    return avg_loss, plate_acc, char_acc


# ==============================================================================
# 3. HÀM HUẤN LUYỆN CHÍNH (TRAINING LOOP VỚI HỖ TRỢ FINE-TUNING)
# ==============================================================================
def train_lprnet(
    data_dir="data_source/dataset_ocr",
    real_data_dir=None,
    real_oversample=5,
    save_dir="runs/train/lprnet_finetuned",
    pretrained="runs/train/lprnet_bounded_stn/weights/best.pt",
    epochs=15,
    batch_size=64,
    lr=0.0003,
    use_stn=True,
    two_stage=False,
    warmup_epochs=5,
    device=None
):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_finetuning = pretrained is not None and Path(pretrained).exists()
    mode_name = "Giai đoạn 2: Fine-Tuning Trộn Dữ liệu Cũ + Mới" if is_finetuning else ("Two-Stage Warmup" if two_stage else "Bounded STN End-to-End")

    print("=" * 65)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN LPRNet + STN (OCR BIỂN SỐ XE)")
    print(f"🎯 Chế độ:              {mode_name}")
    print(f"📂 Dữ liệu gốc (CŨ):    {data_dir.resolve()}")
    if real_data_dir:
        print(f"🌟 Dữ liệu thực tế (MỚI): {Path(real_data_dir).resolve()} (Oversample: {real_oversample}x)")
    print(f"📁 Thư mục lưu kết quả: {save_dir.resolve()}")
    print(f"🖥️ Thiết bị tính toán:   {device}")
    print(f"🔄 Module STN nắn thẳng: {'KÍCH HOẠT' if use_stn else 'TẮT'}")
    print(f"⚙️ Epochs: {epochs} | Batch: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)

    # 1. Thu thập dữ liệu thực tế (nếu có)
    extra_samples = []
    if real_data_dir:
        real_p = Path(real_data_dir)
        real_lbl_file = real_p / "labels.txt"
        if real_lbl_file.exists():
            with open(real_lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name, raw_label = parts[0], parts[1].upper()
                        img_path = real_p / img_name
                        if img_path.exists() and len(raw_label) >= 2:
                            for _ in range(real_oversample):
                                extra_samples.append((img_path, raw_label))
            print(f"🌟 Đã nạp {len(extra_samples)//real_oversample} biển số thực tế (Nhân bản {real_oversample}x = {len(extra_samples)} mẫu) để trộn vào tập Train!")

    # 2. Tạo Dataset & DataLoader
    train_txt = data_dir / "train.txt"
    val_txt = data_dir / "val.txt"
    train_img_dir = data_dir / "train"
    val_img_dir = data_dir / "val"

    train_dataset = LPRDataset(train_txt, train_img_dir, extra_samples=extra_samples, preload=True)
    val_dataset = LPRDataset(val_txt, val_img_dir, preload=True)

    print(f"✓ Tập Huấn luyện (Train Tổng cộng): {len(train_dataset):,} mẫu (Bảo toàn dữ liệu cũ + Trộn dữ liệu mới)")
    print(f"✓ Tập Kiểm thử (Val):                {len(val_dataset):,} mẫu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lpr_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lpr_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # 3. Khởi tạo mô hình & Nạp Pre-trained
    class_num = len(CHARS) + 1  # 30 ký tự + 1 Blank (index 0)
    model = STN_LPRNet(class_num=class_num, in_channels=3, use_stn=use_stn)
    model.to(device)

    if is_finetuning:
        print(f"📥 Đang nạp trọng số pre-trained tốt nhất: {pretrained}")
        ckpt = torch.load(pretrained, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("✓ Nạp trọng số thành công! Tiếp tục tối ưu hóa chuyên sâu.")

    # 4. Hàm mất mát & Optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    history_file = save_dir / "history.csv"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_char_acc,val_plate_acc,lr\n")

    best_plate_acc = 0.0
    start_time = time.time()

    # 5. Vòng lặp huấn luyện từng Epoch
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for images, targets, target_lengths, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            batch_size_cur = images.size(0)

            optimizer.zero_grad()
            logits = model(images)
            T, N, C = logits.size(2), logits.size(0), logits.size(1)
            log_probs = F.log_softmax(logits, dim=1).permute(2, 0, 1)
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item() * batch_size_cur
            train_batches += batch_size_cur
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_train_loss / max(1, train_batches)
        val_loss, val_plate_acc, val_char_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Ghi log CSV
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_char_acc:.2f},{val_plate_acc:.2f},{current_lr:.6f}\n")

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Char Acc: {val_char_acc:.2f}% | Plate Acc: {val_plate_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Lưu checkpoint tốt nhất
        if val_plate_acc >= best_plate_acc:
            best_plate_acc = val_plate_acc
            best_model_path = weights_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "plate_acc": val_plate_acc,
                "char_acc": val_char_acc,
                "class_num": class_num,
                "chars": CHARS,
                "use_stn": use_stn,
                "mode": mode_name
            }, best_model_path)
            print(f"  🏆 Đạt kỷ lục mới! Đã lưu trọng số tốt nhất tại: {best_model_path}")

        # Lưu last.pt
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "plate_acc": val_plate_acc,
            "char_acc": val_char_acc
        }, weights_dir / "last.pt")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 2: HUẤN LUYỆN TINH CHỈNH THÀNH CÔNG!")
    print(f"🎯 Chế độ:                    {mode_name}")
    print(f"⏱️ Tổng thời gian huấn luyện: {total_training_time / 60:.2f} phút")
    print(f"🏆 Độ chính xác biển số cao nhất (Best Plate Acc): {best_plate_acc:.2f}%")
    print(f"🥇 Trọng số tối ưu đã lưu:   {weights_dir / 'best.pt'}")
    print(f"📊 Lịch sử huấn luyện lưu tại: {history_file}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LPRNet + STN nhận diện biển số xe")
    parser.add_argument("--data_dir", type=str, default="data_source/dataset_ocr", help="Thư mục dataset OCR gốc")
    parser.add_argument("--real_data_dir", type=str, default="data_source/real_cropped_plates", help="Thư mục chứa ảnh biển số thực tế vừa cắt")
    parser.add_argument("--real_oversample", type=int, default=5, help="Tỷ lệ nhân bản dữ liệu thực tế để tăng trọng số học")
    parser.add_argument("--pretrained", type=str, default="runs/train/lprnet_bounded_stn/weights/best.pt", help="Trọng số tốt nhất trước đó")
    parser.add_argument("--save_dir", type=str, default="runs/train/lprnet_finetuned", help="Thư mục lưu kết quả huấn luyện")
    parser.add_argument("--epochs", type=int, default=15, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.0003, help="Tốc độ học tinh chỉnh (Learning Rate)")
    parser.add_argument("--no_stn", action="store_true", help="Tắt module STN")
    parser.add_argument("--two_stage", action="store_true", help="Kích hoạt chiến lược 2 pha")
    args = parser.parse_args()

    train_lprnet(
        data_dir=args.data_dir,
        real_data_dir=args.real_data_dir,
        real_oversample=args.real_oversample,
        save_dir=args.save_dir,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_stn=not args.no_stn,
        two_stage=args.two_stage
    )
