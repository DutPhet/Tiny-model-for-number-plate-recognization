"""
==============================================================================
Script Huấn luyện LPRNet + STN (End-to-End với PyTorch CTC Loss)
- Tự động nạp dữ liệu từ data_source/dataset_ocr (10.000 train + 1.500 val).
- Huấn luyện đồng thời SpatialTransformer (nắn thẳng) và LPRNet (đọc ký tự).
- Sử dụng hàm mất mát Connectionist Temporal Classification (CTCLoss).
- Tự động lưu trọng số tốt nhất (best.pt), lịch sử (history.csv) và báo cáo độ chính xác.
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
# 1. DATASET & DATALOADER CHO LPRNET
# ==============================================================================
class LPRDataset(Dataset):
    """
    Dataset đọc ảnh biển số và nhãn text từ file txt:
    Tối ưu hóa In-Memory Caching: Tải toàn bộ ảnh vào RAM (~70MB) giúp tăng tốc 10x trên CPU.
    """
    def __init__(self, txt_path, img_dir, target_size=(94, 24), preload=True):
        self.img_dir = Path(img_dir)
        self.target_w, self.target_h = target_size
        self.samples = []
        self.preload = preload

        if not Path(txt_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, raw_label = parts[0], parts[1].upper()
                    img_path = self.img_dir / img_name
                    if img_path.exists():
                        self.samples.append((img_path, raw_label))

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
    """
    Gộp batch cho PyTorch CTCLoss:
    - imgs: Tensor 4D (Batch_Size, 3, 24, 94)
    - targets: Tensor 1D ghép nối toàn bộ nhãn của cả batch
    - target_lengths: Độ dài chuỗi của từng ảnh trong batch
    - raw_texts: Danh sách chuỗi text gốc để đánh giá Accuracy
    """
    imgs, label_tensors, raw_texts = zip(*batch)
    imgs = torch.stack(imgs, 0)

    target_lengths = torch.tensor([len(t) for t in label_tensors], dtype=torch.long)
    targets = torch.cat(label_tensors) if len(label_tensors) > 0 else torch.empty(0, dtype=torch.long)

    return imgs, targets, target_lengths, raw_texts


# ==============================================================================
# 2. CÁC HÀM TRAIN VÀ ĐÁNH GIÁ (TRAIN & VALIDATE)
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad=5.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="  Train Step", leave=False)
    for imgs, targets, target_lengths, _ in pbar:
        batch_size = imgs.size(0)
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # Forward
        logits = model(imgs)  # (B, class_num, time_steps)
        time_steps = logits.size(2)

        # Chuẩn bị input cho CTCLoss: (Time_Steps, Batch_Size, Class_Num)
        log_probs = F.log_softmax(logits.permute(2, 0, 1), dim=2)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)

        # Tính CTC Loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        # Backward & Optimize
        optimizer.zero_grad()
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        loss_val = loss.item()
        if not np.isnan(loss_val) and not np.isinf(loss_val):
            total_loss += loss_val * batch_size
            total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(1, total_samples)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_plates = 0
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for imgs, targets, target_lengths, raw_texts in tqdm(dataloader, desc="  Val Step  ", leave=False):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(imgs)
            time_steps = logits.size(2)

            log_probs = F.log_softmax(logits.permute(2, 0, 1), dim=2)
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss_val = loss.item()
            if not np.isnan(loss_val) and not np.isinf(loss_val):
                total_loss += loss_val * batch_size
                total_samples += batch_size

            # Giải mã CTC Greedy Decoder
            preds = ctc_decode(logits, CHARS, blank_idx=0)

            # Đánh giá độ chính xác (Plate Accuracy & Char Accuracy)
            for pred_str, gt_str in zip(preds, raw_texts):
                if pred_str == gt_str:
                    correct_plates += 1

                # Đếm ký tự đúng
                total_chars += len(gt_str)
                min_len = min(len(pred_str), len(gt_str))
                for c1, c2 in zip(pred_str[:min_len], gt_str[:min_len]):
                    if c1 == c2:
                        correct_chars += 1

    avg_loss = total_loss / max(1, total_samples)
    plate_acc = (correct_plates / max(1, len(dataloader.dataset))) * 100.0
    char_acc = (correct_chars / max(1, total_chars)) * 100.0

    return avg_loss, plate_acc, char_acc


# ==============================================================================
# 3. QUY TRÌNH HUẤN LUYỆN CHÍNH (MAIN PIPELINE)
# ==============================================================================
def train_lprnet(
    data_dir="data_source/dataset_ocr",
    save_dir="runs/train/lprnet_stn",
    epochs=25,
    batch_size=64,
    lr=0.001,
    use_stn=True,
    two_stage=False,
    warmup_epochs=10,
    device=None
):
    """
    Quy trình huấn luyện hoàn chỉnh mô hình STN + LPRNet:
    - use_stn=True, two_stage=False: Cách 1 - Bounded STN + LPRNet huấn luyện đồng thời từ đầu.
    - use_stn=True, two_stage=True:  Cách 2 - Two-Stage Warmup (đóng băng STN pha 1, tinh chỉnh pha 2).
    """
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_name = "Cách 2: Two-Stage Warmup" if (use_stn and two_stage) else ("Cách 1: Bounded STN End-to-End" if use_stn else "LPRNet Thuần (No STN)")

    print("=" * 65)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN LPRNet + STN (OCR BIỂN SỐ XE)")
    print(f"🎯 Chiến lược:          {mode_name}")
    print(f"📂 Dữ liệu nguồn:        {data_dir.resolve()}")
    print(f"📁 Thư mục lưu kết quả: {save_dir.resolve()}")
    print(f"🖥️ Thiết bị tính toán:   {device}")
    print(f"🔄 Module STN nắn thẳng: {'KÍCH HOẠT' if use_stn else 'TẮT'}")
    if use_stn and two_stage:
        print(f"⏳ Pha 1 Warmup LPRNet:  Epoch 1 -> {warmup_epochs} (Đóng băng STN)")
        print(f"🔥 Pha 2 STN Fine-tune:  Epoch {warmup_epochs + 1} -> {epochs} (Mở khóa STN)")
    print(f"⚙️ Epochs: {epochs} | Batch: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)

    # 1. Tạo Dataset & DataLoader
    train_txt = data_dir / "train.txt"
    val_txt = data_dir / "val.txt"
    train_img_dir = data_dir / "train"
    val_img_dir = data_dir / "val"

    train_dataset = LPRDataset(train_txt, train_img_dir, preload=True)
    val_dataset = LPRDataset(val_txt, val_img_dir, preload=True)

    print(f"✓ Tập Huấn luyện (Train): {len(train_dataset):,} mẫu")
    print(f"✓ Tập Kiểm thử (Val):     {len(val_dataset):,} mẫu")

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

    # 2. Khởi tạo mô hình
    class_num = len(CHARS) + 1  # 30 ký tự + 1 ký tự Blank (index 0)
    model = STN_LPRNet(class_num=class_num, in_channels=3, use_stn=use_stn)
    model.to(device)

    # 3. Hàm mất mát CTC Loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def build_optimizer(cur_epoch):
        """Khởi tạo optimizer phù hợp với từng giai đoạn nếu chạy Two-Stage."""
        if use_stn and two_stage and cur_epoch <= warmup_epochs:
            # Pha 1: Đóng băng STN, chỉ tối ưu LPRNet
            for p in model.stn.parameters():
                p.requires_grad = False
            return torch.optim.AdamW(model.lprnet.parameters(), lr=lr, weight_decay=1e-4)
        elif use_stn and two_stage and cur_epoch > warmup_epochs:
            # Pha 2: Mở khóa STN, STN dùng lr nhỏ hơn (0.1x) để tinh chỉnh êm ái
            for p in model.stn.parameters():
                p.requires_grad = True
            param_groups = [
                {"params": model.lprnet.parameters(), "lr": lr * 0.5},
                {"params": model.stn.parameters(), "lr": lr * 0.1}
            ]
            return torch.optim.AdamW(param_groups, weight_decay=1e-4)
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    optimizer = build_optimizer(1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # File log lịch sử
    history_file = save_dir / "history.csv"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_char_acc,val_plate_acc,lr\n")

    best_plate_acc = 0.0
    start_time = time.time()

    # 4. Vòng lặp huấn luyện từng Epoch
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Kiểm tra chuyển pha nếu chạy Two-Stage
        if use_stn and two_stage and epoch == warmup_epochs + 1:
            print(f"\n🔄 [CHUYỂN PHA HUẤN LUYỆN] Mở khóa Module STN để tinh chỉnh nắn thẳng...")
            optimizer = build_optimizer(epoch)
            remaining_epochs = epochs - warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=lr * 0.005)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_plate_acc, val_char_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Ghi log CSV
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_char_acc:.2f},{val_plate_acc:.2f},{current_lr:.6f}\n")

        # In tiến trình
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Char Acc: {val_char_acc:.2f}% | Plate Acc: {val_plate_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Lưu checkpoint trọng số tốt nhất (best.pt)
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

        # Luôn lưu checkpoint cuối cùng (last.pt)
        last_model_path = weights_dir / "last.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "plate_acc": val_plate_acc,
            "char_acc": val_char_acc
        }, last_model_path)

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print("🎉 HUẤN LUYỆN LPRNet + STN HOÀN TẤT THÀNH CÔNG!")
    print(f"🎯 Chiến lược:                {mode_name}")
    print(f"⏱️ Tổng thời gian huấn luyện: {total_training_time / 60:.2f} phút")
    print(f"🏆 Độ chính xác biển số cao nhất (Best Plate Acc): {best_plate_acc:.2f}%")
    print(f"🥇 Trọng số tối ưu đã lưu:   {weights_dir / 'best.pt'}")
    print(f"📊 Lịch sử huấn luyện lưu tại: {history_file}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LPRNet + STN nhận diện biển số xe")
    parser.add_argument("--data_dir", type=str, default="data_source/dataset_ocr", help="Thư mục dataset OCR")
    parser.add_argument("--save_dir", type=str, default="runs/train/lprnet_stn", help="Thư mục lưu kết quả huấn luyện")
    parser.add_argument("--epochs", type=int, default=25, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học khởi tạo (Learning Rate)")
    parser.add_argument("--no_stn", action="store_true", help="Tắt module STN (huấn luyện LPRNet thuần)")
    parser.add_argument("--two_stage", action="store_true", help="Kích hoạt chiến lược huấn luyện 2 pha (Warmup LPRNet trước)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Số epoch cho pha 1 nếu dùng two_stage")
    args = parser.parse_args()

    train_lprnet(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_stn=not args.no_stn,
        two_stage=args.two_stage,
        warmup_epochs=args.warmup_epochs
    )
