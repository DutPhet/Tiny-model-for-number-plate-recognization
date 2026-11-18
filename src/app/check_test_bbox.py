"""
Script độc lập kiểm tra & vẽ Bounding Box từ `organized_frames_dataset/test`.
Hỗ trợ 2 chế độ:
  1. Mặc định (Ground-Truth Check): Đọc trực tiếp các cặp (ảnh + .txt) trong từng thư mục clip
     của `organized_frames_dataset/test/`, vẽ bounding box nhãn gốc lên ảnh, thống kê lỗi nhãn
     và lưu ảnh kiểm tra (hoặc xem trực tiếp qua cửa sổ OpenCV).
  2. Chế độ Model Check (`--predict` hoặc `--ckpt ...`): Chạy thêm mô hình Module 1 D-FINE lên tập test
     để đối chiếu trực quan giữa Ground Truth (nét liền) và Dự đoán của mô hình (góc nhọn + score).
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_VEHICLE_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

PALETTE_BGR = [
    (255, 190, 50),   # 0: Xanh dương sáng
    (80, 230, 110),   # 1: Xanh lá
    (235, 110, 255),  # 2: Tím hồng
    (65, 135, 255),   # 3: Cam đỏ
    (0, 235, 255),    # 4: Vàng
    (255, 120, 180),  # 5: Hồng
]


def find_project_root() -> Path:
    candidates = []
    if len(HERE.parents) > 1:
        candidates.append(HERE.parents[1])
    candidates.extend([HERE.parent, Path.cwd(), HERE])
    candidates = list(dict.fromkeys(candidates))

    for c in candidates:
        if (c / "src").is_dir() or (c / "data_source").is_dir() or (c / "datasets").is_dir():
            return c
    for c in candidates:
        if (c / "runs").is_dir() and c != HERE:
            return c
    return HERE


def auto_find_test_dir(custom_dir: str = "") -> Path:
    if custom_dir:
        p = Path(custom_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục test: {p.resolve()}")
        return p

    root = find_project_root()
    direct_candidates = [
        root / "data_source" / "organized_frames_dataset" / "test",
        root / "datasets" / "data_source" / "organized_frames_dataset" / "test",
        root / "datasets" / "organized_frames_dataset" / "test",
        root / "datasets" / "data_preparation" / "organized_frames_dataset" / "test",
        root / "organized_frames_dataset" / "test",
        Path.cwd() / "organized_frames_dataset" / "test",
    ]
    for cand in direct_candidates:
        if cand.is_dir():
            return cand

    for base in [root / "data_source", root / "datasets", root]:
        if base.is_dir():
            for match in base.rglob("organized_frames_dataset"):
                if (match / "test").is_dir():
                    return match / "test"

    raise FileNotFoundError(
        "Không tự động tìm thấy thư mục `organized_frames_dataset/test`. "
        "Hãy truyền đường dẫn trực tiếp qua tham số: --test-dir ../../data_source/organized_frames_dataset/test"
    )


def auto_find_m1_ckpt(custom_ckpt: str = "") -> Path:
    if custom_ckpt:
        p = Path(custom_ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {p.resolve()}")
        return p

    root = find_project_root()
    for pat in [
        "runs/sota_alpr/module1_vehicle/*_best.pth",
        "runs/sota_alpr/module1_vehicle/*_last.pth",
        "**/module1_vehicle/*_best.pth",
    ]:
        matches = sorted(root.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError("Không tìm thấy checkpoint Module 1 (.pth). Hãy truyền qua --ckpt.")


def read_image_unicode(img_path: Path):
    data = np.fromfile(str(img_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(out_path: Path, img_bgr: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix if out_path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr)
    if ok:
        buf.tofile(str(out_path))


def parse_label_file(lbl_path: Path, img_w: int, img_h: int):
    records = []
    warnings = []
    if not lbl_path.exists():
        return records, warnings

    text = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return records, warnings

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [tok for tok in re.split(r"[\s,]+", line) if tok]
        if len(parts) < 5:
            warnings.append(f"Dòng {line_no} thiếu cột (<5): '{raw_line}'")
            continue

        raw_cls = parts[0]
        try:
            coords = [float(x) for x in parts[1:5]]
        except ValueError:
            warnings.append(f"Dòng {line_no} lỗi tọa độ số: '{raw_line}'")
            continue

        extra_text = " ".join(parts[5:]) if len(parts) > 5 else ""

        if raw_cls.isdigit():
            cls_id = int(raw_cls)
            cls_label = str(cls_id)
        else:
            cls_id = abs(hash(raw_cls.lower())) % len(PALETTE_BGR)
            cls_label = raw_cls

        c1, c2, c3, c4 = coords
        is_normalized = max(abs(c1), abs(c2), abs(c3), abs(c4)) <= 1.5

        if is_normalized:
            if not (0.0 <= c1 <= 1.0 and 0.0 <= c2 <= 1.0 and 0.0 < c3 <= 1.0 and 0.0 < c4 <= 1.0):
                warnings.append(f"Dòng {line_no} tọa độ YOLO ngoài [0,1]: {coords}")
            x1 = (c1 - 0.5 * c3) * img_w
            y1 = (c2 - 0.5 * c4) * img_h
            x2 = (c1 + 0.5 * c3) * img_w
            y2 = (c2 + 0.5 * c4) * img_h
        else:
            x1, y1, x2, y2 = c1, c2, c3, c4

        x1_c = max(0, min(img_w - 1, int(round(x1))))
        y1_c = max(0, min(img_h - 1, int(round(y1))))
        x2_c = max(0, min(img_w - 1, int(round(x2))))
        y2_c = max(0, min(img_h - 1, int(round(y2))))

        if x2_c <= x1_c or y2_c <= y1_c:
            warnings.append(f"Dòng {line_no} box rỗng/âm sau khi quy đổi: ({x1_c},{y1_c},{x2_c},{y2_c})")
            continue

        records.append({
            "cls_id": cls_id,
            "raw_cls": raw_cls,
            "cls_label": cls_label,
            "xyxy": (x1_c, y1_c, x2_c, y2_c),
            "extra": extra_text,
        })

    return records, warnings


class OptionalModule1Predictor:
    def __init__(self, ckpt_path: Path, conf_thresh: float = 0.45):
        import torch

        for cand in [HERE, HERE.parent, find_project_root() / "src", find_project_root()]:
            if (cand / "models_sota").is_dir() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))

        from models_sota import build_dfine_detector

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.variant = ckpt.get("variant", "dfine_s")
        default_hw = (480, 800) if self.variant == "dfine_l" else (384, 640)
        self.img_size = tuple(ckpt.get("img_size", default_hw))
        num_classes = int(ckpt.get("num_classes", 4))
        self.class_names = ckpt.get("class_names", ["car", "motorcycle", "bus", "truck"])

        self.model = build_dfine_detector(
            num_classes=num_classes,
            variant=self.variant,
            pretrained_backbone=False,
        ).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state"])

        mean = ckpt.get("mean", [0.485, 0.456, 0.406])
        std = ckpt.get("std", [0.229, 0.224, 0.225])
        self.mean_t = torch.tensor(mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std_t = torch.tensor(std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        print(f"🤖 Đã nạp mô hình đối chiếu: {ckpt_path.name} ({self.variant.upper()} trên {self.device})")

    def predict(self, img_bgr: np.ndarray):
        with self.torch.inference_mode():
            h, w = img_bgr.shape[:2]
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            t = self.torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device).float().div_(255.0)
            t.sub_(self.mean_t).div_(self.std_t)

            det = self.model.predict(t, score_thresh=self.conf_thresh)[0]
            boxes = det["boxes"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            scores = det["scores"].cpu().numpy()

            preds = []
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= w / self.img_size[1]
                boxes[:, [1, 3]] *= h / self.img_size[0]
                for box, lbl, sc in zip(boxes, labels, scores):
                    idx0 = int(lbl) - 1 if int(lbl) >= 1 else int(lbl)
                    cname = self.class_names[idx0] if 0 <= idx0 < len(self.class_names) else f"cls_{idx0}"
                    preds.append((tuple(map(int, box)), cname, float(sc)))
            return preds


def draw_frame_overlay(
    img_bgr: np.ndarray,
    clip_name: str,
    frame_name: str,
    gt_records: list,
    has_txt: bool,
    class_map: dict,
    preds: list | None = None,
) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    for rec in gt_records:
        x1, y1, x2, y2 = rec["xyxy"]
        cid = rec["cls_id"]
        color = PALETTE_BGR[cid % len(PALETTE_BGR)]

        if rec["raw_cls"].isdigit():
            cname = class_map.get(cid, f"cls_{cid}")
        else:
            cname = rec["cls_label"]

        tag = f"GT:{cname}"
        if rec["extra"]:
            tag += f" [{rec['extra']}]"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(vis, (x1, y_top), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, tag, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 2, cv2.LINE_AA)

    if preds is not None:
        for (px1, py1, px2, py2), cname, score in preds:
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w - 1, px2), min(h - 1, py2)
            p_color = (0, 255, 255)
            cv2.rectangle(vis, (px1, py1), (px2, py2), p_color, 1)
            c_len = max(8, min((px2 - px1) // 5, (py2 - py1) // 5))
            for cx, cy, dx, dy in [
                (px1, py1, 1, 1), (px2, py1, -1, 1),
                (px1, py2, 1, -1), (px2, py2, -1, -1),
            ]:
                cv2.line(vis, (cx, cy), (cx + dx * c_len, cy), p_color, 3)
                cv2.line(vis, (cx, cy), (cx, cy + dy * c_len), p_color, 3)

            ptag = f"PRED:{cname} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(ptag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            by2 = min(h - 1, py2 + th + 8)
            cv2.rectangle(vis, (px1, py2), (px1 + tw + 8, by2), (20, 20, 20), -1)
            cv2.putText(vis, ptag, (px1 + 4, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, p_color, 1, cv2.LINE_AA)

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    txt_status = f"GT Boxes: {len(gt_records)}" if has_txt else "NO .TXT LABEL"
    if preds is not None:
        txt_status += f" | Pred: {len(preds)}"
    header = f"{clip_name} / {frame_name}  ({w}x{h})  |  {txt_status}"
    cv2.putText(vis, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 190), 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra & vẽ Bounding Box tập test của organized_frames_dataset")
    parser.add_argument("--test-dir", type=str, default="", help="Đường dẫn tới `organized_frames_dataset/test`")
    parser.add_argument("--out-dir", type=str, default="", help="Thư mục xuất ảnh đã vẽ box kiểm tra")
    parser.add_argument("--classes", type=str, default="", help="Tên lớp cách nhau dấu phẩy (vd: 'car,motorcycle,bus,truck' hoặc 'plate')")
    parser.add_argument("--max-clips", type=int, default=0, help="Giới hạn số thư mục clip kiểm tra (0 = chạy hết)")
    parser.add_argument("--max-per-clip", type=int, default=0, help="Giới hạn số frame mỗi clip (0 = chạy hết)")
    parser.add_argument("--predict", action="store_true", help="Chạy thêm model Module 1 D-FINE để đối chiếu với GT")
    parser.add_argument("--ckpt", type=str, default="", help="Đường dẫn checkpoint .pth khi dùng --predict")
    parser.add_argument("--conf-thresh", type=float, default=0.45, help="Ngưỡng confidence khi dùng --predict")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ xem trực tiếp từng frame (A/D chuyển frame, N chuyển clip, Q thoát)")
    parser.add_argument("--no-save", action="store_true", help="Không lưu ảnh ra ổ cứng (chỉ thống kê / xem qua --show)")
    args = parser.parse_args()

    test_dir = auto_find_test_dir(args.test_dir)
    root = find_project_root()
    out_dir = Path(args.out_dir) if args.out_dir else (root / "runs" / "check_organized_test")

    clip_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if not clip_dirs:
        clip_dirs = [test_dir]

    if args.max_clips > 0:
        clip_dirs = clip_dirs[:args.max_clips]

    predictor = None
    if args.predict or args.ckpt:
        ckpt_path = auto_find_m1_ckpt(args.ckpt)
        predictor = OptionalModule1Predictor(ckpt_path, conf_thresh=args.conf_thresh)

    samples = []
    class_counter = Counter()
    all_warnings = []
    missing_txt_count = 0
    empty_txt_count = 0

    for cdir in clip_dirs:
        imgs = sorted([p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if args.max_per_clip > 0:
            imgs = imgs[:args.max_per_clip]
        for img_p in imgs:
            txt_p = img_p.with_suffix(".txt")
            samples.append((cdir, img_p, txt_p))

    if not samples:
        raise RuntimeError(f"Không tìm thấy file ảnh nào trong: {test_dir}")

    if args.classes.strip():
        custom_list = [c.strip() for c in args.classes.split(",") if c.strip()]
        class_map = {i: name for i, name in enumerate(custom_list)}
    else:
        class_map = dict(DEFAULT_VEHICLE_NAMES)

    print(f"📂 Thư mục test : {test_dir}")
    print(f"🎬 Số thư mục clip: {len(clip_dirs)} | Tổng số frame kiểm tra: {len(samples)}")
    if not args.no_save:
        print(f"💾 Ảnh vẽ box sẽ lưu tại: {out_dir}")
    if args.show:
        print("⌨️  Phím tắt xem trực tiếp: [D / Space] Tiếp | [A] Lùi | [N] Qua clip kế | [Q / ESC] Thoát")

    rendered_items = []
    for idx, (cdir, img_p, txt_p) in enumerate(samples, 1):
        img = read_image_unicode(img_p)
        if img is None:
            all_warnings.append(f"Không đọc được ảnh: {img_p}")
            continue

        h, w = img.shape[:2]
        has_txt = txt_p.exists()
        if not has_txt:
            missing_txt_count += 1
            gt_records, warns = [], []
        else:
            gt_records, warns = parse_label_file(txt_p, w, h)
            if len(gt_records) == 0:
                empty_txt_count += 1
            for w_msg in warns:
                all_warnings.append(f"[{cdir.name}/{txt_p.name}] {w_msg}")

        for rec in gt_records:
            lbl_key = class_map.get(rec["cls_id"], rec["cls_label"]) if rec["raw_cls"].isdigit() else rec["cls_label"]
            class_counter[lbl_key] += 1

        preds = predictor.predict(img) if predictor is not None else None
        vis = draw_frame_overlay(
            img_bgr=img,
            clip_name=cdir.name if cdir != test_dir else "test",
            frame_name=img_p.name,
            gt_records=gt_records,
            has_txt=has_txt,
            class_map=class_map,
            preds=preds,
        )

        if not args.no_save:
            rel_folder = cdir.name if cdir != test_dir else ""
            save_path = out_dir / rel_folder / f"bbox_{img_p.name}"
            write_image_unicode(save_path, vis)

        if args.show:
            rendered_items.append((cdir.name, img_p.name, vis))

        if idx % 100 == 0 or idx == len(samples):
            print(f"   Đã xử lý {idx}/{len(samples)} frames...", end="\r")

    print("\n\n📊 KẾT QUẢ KIỂM TRA TẬP TEST:")
    print(f"   • Tổng frame đã quét         : {len(samples)}")
    print(f"   • Frame thiếu file .txt      : {missing_txt_count}")
    print(f"   • Frame có .txt nhưng rỗng   : {empty_txt_count}")
    print(f"   • Tổng số Bounding Box GT    : {sum(class_counter.values())}")
    for cname, cnt in sorted(class_counter.items()):
        print(f"      - {cname:<14}: {cnt} boxes")

    if all_warnings:
        print(f"\n⚠️ Phát hiện {len(all_warnings)} cảnh báo nhãn (hiển thị tối đa 15 dòng đầu):")
        for w_line in all_warnings[:15]:
            print(f"   • {w_line}")
    else:
        print("\n✅ Không phát hiện lỗi định dạng hay tọa độ vượt khung nào trong các file .txt!")

    if args.show and rendered_items:
        cur = 0
        win_name = "Check Organized Test BBox [A:Prev | D/Space:Next | N:Next Clip | Q:Quit]"
        while 0 <= cur < len(rendered_items):
            clip_n, _, vis_img = rendered_items[cur]
            cv2.imshow(win_name, vis_img)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("a"), ord("A"), 81):
                cur = max(0, cur - 1)
            elif key in (ord("n"), ord("N")):
                nxt = cur + 1
                while nxt < len(rendered_items) and rendered_items[nxt][0] == clip_n:
                    nxt += 1
                cur = nxt if nxt < len(rendered_items) else len(rendered_items) - 1
            else:
                cur += 1
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()