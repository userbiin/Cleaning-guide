"""
Working Desk + LVIS 통합 파인튜닝 데이터셋 빌드 — datasets/build_merged.py

실행:
  python datasets/build_merged.py
  python datasets/build_merged.py --lvis-samples 2000   # 빠른 테스트
  python datasets/build_merged.py --skip-lvis            # Working Desk만
  python datasets/build_merged.py --rebuild              # 기존 merged 초기화 후 재빌드
"""

import argparse
import os
import random
import shutil
import stat
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# ── 통합 클래스 ───────────────────────────────────────────────────────────────
UNIFIED_CLASSES = [
    "book",          # 0
    "pen",           # 1
    "pencil",        # 2
    "scissors",      # 3
    "knife",         # 4
    "laptop",        # 5
    "keyboard",      # 6
    "mouse",         # 7
    "phone",         # 8
    "remote_control",# 9
    "cup",           # 10
    "bottle",        # 11
    "plate",         # 12
    "bowl",          # 13
    "spoon",         # 14
    "fork",          # 15
    "backpack",      # 16
    "bag",           # 17
    "box",           # 18
    "toothbrush",    # 19
    "cable",         # 20
    "cosmetics",     # 21
    "comb",          # 22
    "mirror",        # 23
]
CLASS_TO_IDX = {c: i for i, c in enumerate(UNIFIED_CLASSES)}

# ── Working Desk 클래스 인덱스 → 통합 인덱스 (직접 매핑) ─────────────────────
# data.yaml names: [bottle(0), cell phone(1), cup(2), cutter(3), fork(4),
#                   keyboard(5), knife(6), mouse(7), remote(8), scissors(9), spoon(10)]
DESK_IDX_TO_UNIFIED: dict[int, int] = {
    0:  CLASS_TO_IDX["bottle"],        # bottle      → 11
    1:  CLASS_TO_IDX["phone"],         # cell phone  → 8
    2:  CLASS_TO_IDX["cup"],           # cup         → 10
    3:  CLASS_TO_IDX["scissors"],      # cutter      → 3
    4:  CLASS_TO_IDX["fork"],          # fork        → 15
    5:  CLASS_TO_IDX["keyboard"],      # keyboard    → 6
    6:  CLASS_TO_IDX["knife"],         # knife       → 4
    7:  CLASS_TO_IDX["mouse"],         # mouse       → 7
    8:  CLASS_TO_IDX["remote_control"],# remote      → 9
    9:  CLASS_TO_IDX["scissors"],      # scissors    → 3
    10: CLASS_TO_IDX["spoon"],         # spoon       → 14
}

# ── LVIS 클래스 → 통합 인덱스 ────────────────────────────────────────────────
LVIS_TO_UNIFIED: dict[str, int] = {
    "book":                      CLASS_TO_IDX["book"],
    "pen":                       CLASS_TO_IDX["pen"],
    "pencil":                    CLASS_TO_IDX["pencil"],
    "scissors":                  CLASS_TO_IDX["scissors"],
    "knife":                     CLASS_TO_IDX["knife"],         # ← 추가
    "laptop_computer":           CLASS_TO_IDX["laptop"],
    "computer_keyboard":         CLASS_TO_IDX["keyboard"],
    "mouse_(computer_equipment)":CLASS_TO_IDX["mouse"],         # ← 추가
    "computer_mouse":            CLASS_TO_IDX["mouse"],
    "cellular_telephone":        CLASS_TO_IDX["phone"],
    "remote_control":            CLASS_TO_IDX["remote_control"],
    "cup":                       CLASS_TO_IDX["cup"],
    "bottle":                    CLASS_TO_IDX["bottle"],
    "plate":                     CLASS_TO_IDX["plate"],
    "bowl":                      CLASS_TO_IDX["bowl"],
    "spoon":                     CLASS_TO_IDX["spoon"],
    "fork":                      CLASS_TO_IDX["fork"],
    "backpack":                  CLASS_TO_IDX["backpack"],
    "handbag":                   CLASS_TO_IDX["bag"],
    "box":                       CLASS_TO_IDX["box"],
    "toothbrush":                CLASS_TO_IDX["toothbrush"],
    "power_cord":                CLASS_TO_IDX["cable"],
    "electrical_cable":          CLASS_TO_IDX["cable"],
    "lipstick":                  CLASS_TO_IDX["cosmetics"],
    "hairbrush":                 CLASS_TO_IDX["comb"],
    "comb":                      CLASS_TO_IDX["comb"],
    "mirror":                    CLASS_TO_IDX["mirror"],
}

# LVIS 클래스별 최대 샘플 수
LVIS_CLASS_QUOTA: dict[str, int] = {
    "book": 500, "pen": 200, "pencil": 200,
    "knife": 150, "laptop_computer": 300, "plate": 200, "bowl": 200,
    "backpack": 200, "handbag": 200, "box": 200,
    "toothbrush": 100, "power_cord": 200, "electrical_cable": 200,
    "lipstick": 200, "hairbrush": 150, "comb": 150, "mirror": 150,
    "cup": 150, "bottle": 150, "spoon": 100, "fork": 100,
    "scissors": 100, "computer_keyboard": 150,
    "mouse_(computer_equipment)": 100, "computer_mouse": 100,
    "cellular_telephone": 150, "remote_control": 100,
}
LVIS_TARGET_CLASSES = set(LVIS_CLASS_QUOTA.keys())

# ── 경로 ─────────────────────────────────────────────────────────────────────
DESK_DIR   = _ROOT / "eval" / "data" / "desk"
MERGED_DIR = _ROOT / "datasets" / "merged"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _polygon_to_bbox_yolo(parts: list[str]) -> tuple[int, float, float, float, float] | None:
    """polygon 또는 bbox 포맷 → (cls_idx, cx, cy, w, h) 정규화"""
    if len(parts) < 5:
        return None
    cls_idx = int(parts[0])
    coords  = list(map(float, parts[1:]))
    if len(coords) > 4:          # polygon
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w  = x_max - x_min
        h  = y_max - y_min
    else:                        # already bbox
        cx, cy, w, h = coords
    return cls_idx, cx, cy, w, h


def _save_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── 2단계: Working Desk 변환 ──────────────────────────────────────────────────

def convert_desk(stats: dict) -> None:
    print("\n[2단계] Working Desk 변환 중...")

    # train→train, valid→val, test→val
    splits = [("train", "train"), ("valid", "val"), ("test", "val")]

    for split, dst_split in splits:
        img_src = DESK_DIR / split / "images"
        lbl_src = DESK_DIR / split / "labels"
        img_dst = MERGED_DIR / "images" / dst_split
        lbl_dst = MERGED_DIR / "labels" / dst_split

        if not img_src.exists():
            print(f"  [스킵] {split} 디렉터리 없음")
            continue

        img_paths = sorted(list(img_src.glob("*.jpg")) + list(img_src.glob("*.png")))
        converted = 0

        for img_path in img_paths:
            lbl_path = lbl_src / (img_path.stem + ".txt")
            rows: list[str] = []

            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parsed = _polygon_to_bbox_yolo(line.split())
                    if parsed is None:
                        continue
                    old_idx, cx, cy, w, h = parsed
                    new_idx = DESK_IDX_TO_UNIFIED.get(old_idx)
                    if new_idx is None:
                        continue
                    rows.append(f"{new_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats["desk"][UNIFIED_CLASSES[new_idx]] += 1

            if rows:
                _copy_image(img_path, img_dst / img_path.name)
                _save_label(lbl_dst / (img_path.stem + ".txt"), rows)
                converted += 1

        print(f"  {split:5s} → {dst_split}: {converted}장 변환 완료")


# ── 3단계: LVIS 다운로드 + 변환 ──────────────────────────────────────────────

def convert_lvis(max_samples: int, stats: dict) -> None:
    print(f"\n[3단계] LVIS 다운로드 (max_samples={max_samples})...")
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    dataset_name = f"lvis_hf_{max_samples}_merged"
    if dataset_name in fo.list_datasets():
        print(f"  기존 캐시 로드: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = fouh.load_from_hub(
            "Voxel51/LVIS",
            max_samples=max_samples,
            name=dataset_name,
            persistent=True,
        )

    # GT 필드 탐지
    sample0 = dataset.first()
    gt_field = next(
        (f for f in sample0.field_names if hasattr(sample0.get_field(f), "detections")),
        None,
    )
    if gt_field is None:
        print("  [오류] Detections 필드 없음")
        return

    img_dst = MERGED_DIR / "images" / "train"
    lbl_dst = MERGED_DIR / "labels" / "train"

    # 클래스별 quota 추적
    quota_used: dict[str, int] = defaultdict(int)
    added_ids: set[str] = set()
    total_added = 0

    for sample in dataset.iter_samples():
        if sample.id in added_ids:
            continue

        ann = sample.get_field(gt_field)
        if not ann or not ann.detections:
            continue

        rows: list[str] = []
        for det in ann.detections:
            lvis_label = det.label
            unified_idx = LVIS_TO_UNIFIED.get(lvis_label)
            if unified_idx is None:
                continue

            # quota 확인
            if quota_used[lvis_label] >= LVIS_CLASS_QUOTA.get(lvis_label, 0):
                continue

            bx, by, bw, bh = det.bounding_box
            cx = bx + bw / 2
            cy = by + bh / 2
            # 유효성 검사
            if bw <= 0 or bh <= 0:
                continue

            rows.append(f"{unified_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["lvis"][UNIFIED_CLASSES[unified_idx]] += 1
            quota_used[lvis_label] += 1

        if rows:
            stem = f"lvis_{sample.id}"
            dst_img = img_dst / (stem + Path(sample.filepath).suffix)
            _copy_image(Path(sample.filepath), dst_img)
            _save_label(lbl_dst / (stem + ".txt"), rows)
            added_ids.add(sample.id)
            total_added += 1

        # 모든 quota 달성 시 조기 종료
        all_done = all(
            quota_used[cls] >= LVIS_CLASS_QUOTA[cls]
            for cls in LVIS_CLASS_QUOTA
            if LVIS_CLASS_QUOTA[cls] > 0
        )
        if all_done:
            print(f"  모든 quota 달성 → 조기 종료")
            break

    print(f"  LVIS → train: {total_added}장 추가")
    print(f"  클래스별 수집량: {dict(quota_used)}")


# ── 4단계: data.yaml 생성 ─────────────────────────────────────────────────────

def write_yaml() -> None:
    data = {
        "path":  str(MERGED_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"\n[4단계] data.yaml 저장: {yaml_path}")


# ── 5단계: 검증 ───────────────────────────────────────────────────────────────

def verify(stats: dict) -> None:
    from PIL import Image, ImageDraw

    print("\n[5단계] 검증...")

    train_imgs = list((MERGED_DIR / "images" / "train").glob("*.jpg")) + \
                 list((MERGED_DIR / "images" / "train").glob("*.png"))
    val_imgs   = list((MERGED_DIR / "images" / "val").glob("*.jpg")) + \
                 list((MERGED_DIR / "images" / "val").glob("*.png"))

    SEP = "=" * 60
    lines = [
        SEP,
        "  통합 데이터셋 검증 결과",
        SEP,
        f"(1) 이미지 수",
        f"    train: {len(train_imgs)}장",
        f"    val:   {len(val_imgs)}장",
        f"    합계:  {len(train_imgs)+len(val_imgs)}장",
        "",
        "(2) 클래스별 annotation 수 (히스토그램)",
        f"    {'클래스':<18} {'Working Desk':>14} {'LVIS':>8} {'합계':>8}",
        f"    {'-'*52}",
    ]

    total_desk = sum(stats["desk"].values())
    total_lvis = sum(stats["lvis"].values())
    all_counts: dict[str, int] = defaultdict(int)
    for cls in UNIFIED_CLASSES:
        d = stats["desk"].get(cls, 0)
        l = stats["lvis"].get(cls, 0)
        t = d + l
        all_counts[cls] = t
        bar = "#" * min(30, t // 20)
        lines.append(f"    {cls:<18} {d:>14,} {l:>8,} {t:>8,}  {bar}")

    lines += [
        f"    {'-'*52}",
        f"    {'합계':<18} {total_desk:>14,} {total_lvis:>8,} {total_desk+total_lvis:>8,}",
        "",
        f"(3) 기여분",
        f"    Working Desk: {total_desk:,}개 ({total_desk/(total_desk+total_lvis)*100:.1f}%)",
        f"    LVIS:         {total_lvis:,}개 ({total_lvis/(total_desk+total_lvis)*100:.1f}%)" if total_lvis else "    LVIS: 0 (스킵됨)",
        "",
        "(4) 클래스 불균형 경고",
    ]

    counts_list = [(c, all_counts[c]) for c in UNIFIED_CLASSES if all_counts[c] > 0]
    if counts_list:
        max_c = max(counts_list, key=lambda x: x[1])
        min_c = min(counts_list, key=lambda x: x[1])
        ratio = max_c[1] / max(min_c[1], 1)
        lines.append(f"    최다: {max_c[0]} ({max_c[1]:,}개)  최소: {min_c[0]} ({min_c[1]:,}개)")
        lines.append(f"    불균형 비율: {ratio:.1f}x")
        if ratio > 10:
            lines.append(f"    [!] 불균형 심각 (>10x) -- 오버샘플링 또는 가중치 조정 권장")
        elif ratio > 5:
            lines.append(f"    [!] 불균형 주의 (>5x)")
        else:
            lines.append(f"    [OK] 균형 양호")
    else:
        lines.append("    데이터 없음")

    lines.append(SEP)
    summary = "\n".join(lines)
    enc = sys.stdout.encoding or "utf-8"
    print(summary.encode(enc, errors="replace").decode(enc))

    # 검증 이미지 저장
    verify_dir = MERGED_DIR / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = train_imgs + val_imgs
    samples = random.sample(all_imgs, min(3, len(all_imgs)))

    print("\n(5) 검증 이미지:")
    for img_path in samples:
        lbl_path_train = MERGED_DIR / "labels" / "train" / (img_path.stem + ".txt")
        lbl_path_val   = MERGED_DIR / "labels" / "val"   / (img_path.stem + ".txt")
        lbl_path = lbl_path_train if lbl_path_train.exists() else lbl_path_val

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        draw = ImageDraw.Draw(img)

        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                cls_name = UNIFIED_CLASSES[cls_idx] if cls_idx < len(UNIFIED_CLASSES) else str(cls_idx)
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1 + 2, y1 + 2), cls_name, fill="lime")

        out = verify_dir / img_path.name
        img.save(out)
        print(f"    저장: {out.name}")

    (verify_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\n검증 요약: {verify_dir}/summary.txt")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Working Desk + LVIS 통합 데이터셋 빌드")
    p.add_argument("--lvis-samples", type=int, default=5000,
                   help="LVIS에서 로드할 최대 샘플 수 (기본: 5000)")
    p.add_argument("--skip-lvis", action="store_true",
                   help="LVIS 다운로드 스킵 (Working Desk만)")
    p.add_argument("--rebuild", action="store_true",
                   help="기존 merged 디렉터리 초기화 후 재빌드")
    args = p.parse_args()

    def _rm_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # 출력 디렉터리 초기화 (--rebuild 시 기존 파일 제거)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = MERGED_DIR / sub
        if args.rebuild and d.exists():
            shutil.rmtree(d, onexc=_rm_readonly)
        d.mkdir(parents=True, exist_ok=True)

    print(f"통합 클래스 ({len(UNIFIED_CLASSES)}개): {UNIFIED_CLASSES}")
    print(f"Working Desk 인덱스 매핑: {DESK_IDX_TO_UNIFIED}")

    stats: dict[str, dict[str, int]] = {
        "desk": defaultdict(int),
        "lvis": defaultdict(int),
    }

    # 2단계: Working Desk 변환
    convert_desk(stats)

    # 3단계: LVIS 변환 (옵션)
    if not args.skip_lvis:
        convert_lvis(args.lvis_samples, stats)
    else:
        print("\n[3단계] LVIS 스킵")

    # 4단계: data.yaml
    write_yaml()

    # 5단계: 검증
    verify(stats)


if __name__ == "__main__":
    main()
