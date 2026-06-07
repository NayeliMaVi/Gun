from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

from utils_paths import dataset_source_paths, resolve_project_root, workspace_paths
from utils_voc import (
    YOLO_CLASS_NAME,
    list_images,
    materialize_file,
    parse_voc_xml,
    safe_stem,
    voc_to_yolo_line,
    write_label_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara un dataset YOLO externo sin modificar detector/.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--copy-mode", choices=("copy", "link", "symlink"), default="link")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-pos-train", type=int, default=None)
    parser.add_argument("--max-pos-val", type=int, default=None)
    parser.add_argument("--max-pos-test", type=int, default=None)
    parser.add_argument("--max-neg-train", type=int, default=None)
    parser.add_argument("--max-neg-val", type=int, default=None)
    parser.add_argument("--max-neg-test", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def reset_dataset_dir(dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_positive_pairs(image_dir: Path, xml_dir: Path) -> List[Tuple[Path, Path]]:
    image_by_stem = {path.stem.lower(): path for path in list_images(image_dir)}
    pairs: List[Tuple[Path, Path]] = []
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        annotation = parse_voc_xml(xml_path)
        image_path = image_by_stem.get(xml_path.stem.lower()) or image_by_stem.get(Path(annotation.filename).stem.lower())
        if image_path is not None:
            pairs.append((image_path, xml_path))
    return pairs


def split_train_val(
    pairs: Sequence[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    items = list(pairs)
    random.Random(seed).shuffle(items)
    val_count = int(len(items) * val_ratio)
    if len(items) > 1:
        val_count = max(1, val_count)
    return items[val_count:], items[:val_count]


def apply_limit(items: Sequence, limit: int | None) -> List:
    items = list(items)
    return items if limit is None else items[:limit]


def export_positive_split(
    items: Sequence[Tuple[Path, Path]],
    split: str,
    dataset_root: Path,
    copy_mode: str,
) -> Dict[str, int]:
    stats = {"images": 0, "labels": 0, "errors": 0}
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    for image_path, xml_path in tqdm(items, desc=f"Positivos {split}", unit="img"):
        try:
            annotation = parse_voc_xml(xml_path)
            lines = []
            for obj in annotation.objects:
                if obj.class_name and obj.class_name != YOLO_CLASS_NAME:
                    continue
                lines.append(voc_to_yolo_line(obj, annotation.width, annotation.height))
            if not lines:
                stats["errors"] += 1
                continue

            stem = safe_stem(f"gun_{split}", image_path)
            output_image = image_dir / f"{stem}{image_path.suffix.lower()}"
            output_label = label_dir / f"{stem}.txt"
            materialize_file(image_path, output_image, copy_mode)
            write_label_file(output_label, lines)
            stats["images"] += 1
            stats["labels"] += 1
        except Exception:
            stats["errors"] += 1

    return stats


def export_negative_split(
    images: Sequence[Path],
    split: str,
    dataset_root: Path,
    copy_mode: str,
) -> Dict[str, int]:
    stats = {"images": 0, "labels": 0, "errors": 0}
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    for image_path in tqdm(images, desc=f"Negativos {split}", unit="img"):
        try:
            stem = safe_stem(f"other_{split}", image_path)
            output_image = image_dir / f"{stem}{image_path.suffix.lower()}"
            output_label = label_dir / f"{stem}.txt"
            materialize_file(image_path, output_image, copy_mode)
            write_label_file(output_label, [])
            stats["images"] += 1
            stats["labels"] += 1
        except Exception:
            stats["errors"] += 1

    return stats


def write_data_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: gun",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    source = dataset_source_paths(project_root)
    workspace = workspace_paths(project_root)
    dataset_root = workspace["yolo_dataset"]

    if args.clean:
        reset_dataset_dir(dataset_root)
    else:
        for split in ("train", "val", "test"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_pairs = build_positive_pairs(source["gun_train_images"], source["gun_train_xml"])
    train_pairs, val_pairs = split_train_val(train_pairs, args.val_ratio, args.seed)
    test_pairs = build_positive_pairs(source["gun_test_images"], source["gun_test_xml"])

    train_pairs = apply_limit(train_pairs, args.max_pos_train)
    val_pairs = apply_limit(val_pairs, args.max_pos_val)
    test_pairs = apply_limit(test_pairs, args.max_pos_test)

    negative_images = list_images(source["other_images"])
    random.Random(args.seed).shuffle(negative_images)
    neg_train = apply_limit(negative_images, args.max_neg_train)
    neg_val = apply_limit(negative_images[len(neg_train):], args.max_neg_val)
    neg_test = apply_limit(negative_images[len(neg_train) + len(neg_val):], args.max_neg_test)

    stats = {
        "pos_train": export_positive_split(train_pairs, "train", dataset_root, args.copy_mode),
        "pos_val": export_positive_split(val_pairs, "val", dataset_root, args.copy_mode),
        "pos_test": export_positive_split(test_pairs, "test", dataset_root, args.copy_mode),
        "neg_train": export_negative_split(neg_train, "train", dataset_root, args.copy_mode),
        "neg_val": export_negative_split(neg_val, "val", dataset_root, args.copy_mode),
        "neg_test": export_negative_split(neg_test, "test", dataset_root, args.copy_mode),
    }
    yaml_path = write_data_yaml(dataset_root)

    print("Dataset YOLO preparado en:")
    print(f"  - {dataset_root}")
    print(f"  - data.yaml: {yaml_path}")
    for key, value in stats.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
