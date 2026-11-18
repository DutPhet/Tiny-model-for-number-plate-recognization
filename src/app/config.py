import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_VEHICLE_CLASSES = {
    1: "CAR",
    2: "MOTO",
    3: "BUS",
    4: "TRUCK",
}

CLASS_COLORS = {
    1: (255, 190, 50),   # CAR
    2: (80, 230, 110),   # MOTO
    3: (235, 110, 255),  # BUS
    4: (65, 135, 255),   # TRUCK
}


def resolve_project_paths() -> tuple[Path, Path]:
    # Ưu tiên thư mục gốc dự án (cha của src/) để tránh nhận nhầm thư mục con src/app/runs/
    candidates = []
    if len(APP_DIR.parents) > 1:
        candidates.append(APP_DIR.parents[1])
    candidates.extend([APP_DIR.parent, Path.cwd(), APP_DIR])
    candidates = list(dict.fromkeys(candidates))

    src_dir = APP_DIR.parent
    for c in candidates:
        if (c / "models_sota").is_dir():
            src_dir = c
            break

    project_root = APP_DIR.parents[1] if len(APP_DIR.parents) > 1 else APP_DIR.parent
    for c in candidates:
        if (c / "runs" / "sota_alpr").is_dir() or (c / "datasets").is_dir():
            project_root = c
            break
    else:
        for c in candidates:
            if (c / "runs").is_dir() and c != APP_DIR:
                project_root = c
                break

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return src_dir, project_root


def auto_find_m1_checkpoint(custom_path: str = "") -> Path:
    if custom_path:
        p = Path(custom_path)
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy file trọng số Module 1: {p.resolve()}")
        return p

    src_dir, project_root = resolve_project_paths()
    extra_roots = [APP_DIR.parents[1]] if len(APP_DIR.parents) > 1 else []
    search_roots = list(dict.fromkeys([project_root, *extra_roots, src_dir, Path.cwd()]))
    patterns = [
        "runs/sota_alpr/module1_vehicle/*_best.pth",
        "runs/sota_alpr/module1_vehicle/*_last.pth",
        "runs/*/module1_vehicle/*_best.pth",
        "runs/*/module1_vehicle/*_last.pth",
        "**/module1_vehicle/*.pth",
    ]

    for root in search_roots:
        for pat in patterns:
            matches = sorted(root.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    raise FileNotFoundError(
        "Không tìm thấy checkpoint Module 1 (module1_vehicle/*_best.pth). "
        "Hãy truyền đường dẫn trực tiếp qua tham số: --ckpt ../../runs/sota_alpr/module1_vehicle/module1_vehicle_dfine_s_best.pth"
    )