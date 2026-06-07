from __future__ import annotations

import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image
from tqdm import tqdm


SEED = 42
IMAGE_SUFFIX = ".jpg"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def list_jpgs(directory: Path) -> List[Path]:
    return sorted(path for path in directory.rglob(f"*{IMAGE_SUFFIX}") if path.is_file())


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def pick_split(files: Sequence[Path], train_count: int, val_count: int, rng: random.Random) -> Dict[str, List[Path]]:
    items = list(files)
    rng.shuffle(items)
    train = items[: min(train_count, len(items))]
    remaining = items[len(train) :]
    val = remaining[: min(val_count, len(remaining))]
    return {"train": train, "val": val}


def pick_test(files: Sequence[Path], test_count: int, rng: random.Random) -> List[Path]:
    items = list(files)
    rng.shuffle(items)
    return items[: min(test_count, len(items))]


def ensure_dirs(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        for label in ("arma", "no_arma"):
            (dataset_root / split / label).mkdir(parents=True, exist_ok=True)


def copy_images(files: Iterable[Path], destination_dir: Path, prefix: str) -> Dict[str, int]:
    items = list(files)
    stats = {"requested": 0, "copied": 0, "invalid": 0, "skipped": 0}
    existing_count = count_jpgs(destination_dir)
    if existing_count >= len(items):
        stats["requested"] = len(items)
        stats["skipped"] = len(items)
        return stats

    def _copy_one(payload: tuple[int, Path]) -> str:
        index, source = payload
        destination_name = f"{prefix}_{index:05d}{source.suffix.lower()}"
        destination = destination_dir / destination_name
        if destination.exists():
            return "skipped"
        shutil.copyfile(source, destination)
        if not verify_image(destination):
            if destination.exists():
                destination.unlink()
            return "invalid"
        return "copied"

    with ThreadPoolExecutor(max_workers=8) as executor:
        iterator = executor.map(_copy_one, enumerate(items, start=1))
        for result in tqdm(iterator, total=len(items), desc=f"Copiando {destination_dir.relative_to(project_root())}", unit="img"):
            stats["requested"] += 1
            stats[result] += 1
    return stats


def count_jpgs(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*.jpg") if path.is_file())


def main() -> None:
    root = project_root()
    rng = random.Random(SEED)

    dataset_root = root / "workspace_modelos" / "dataset_clasificacion_nasnet"
    ensure_dirs(dataset_root)

    positive_train_src = root / "classifier" / "gun" / "train"
    positive_test_src = root / "classifier" / "gun" / "test"
    negative_train_src = root / "classifier" / "other" / "other" / "train"
    negative_test_src = root / "classifier" / "other" / "other" / "test"

    positive_train_files = list_jpgs(positive_train_src)
    positive_test_files = list_jpgs(positive_test_src)
    negative_train_files = list_jpgs(negative_train_src)
    negative_test_files = list_jpgs(negative_test_src)

    positive_split = pick_split(positive_train_files, train_count=20000, val_count=5000, rng=rng)
    negative_split = pick_split(negative_train_files, train_count=20000, val_count=5000, rng=rng)
    positive_test = pick_test(positive_test_files, test_count=1000, rng=rng)
    negative_test = pick_test(negative_test_files, test_count=1000, rng=rng)

    availability_warnings: List[str] = []
    if len(positive_split["train"]) < 20000:
        availability_warnings.append(f"arma train: solo {len(positive_split['train'])} disponibles")
    if len(positive_split["val"]) < 5000:
        availability_warnings.append(f"arma val: solo {len(positive_split['val'])} disponibles")
    if len(negative_split["train"]) < 20000:
        availability_warnings.append(f"no_arma train: solo {len(negative_split['train'])} disponibles")
    if len(negative_split["val"]) < 5000:
        availability_warnings.append(f"no_arma val: solo {len(negative_split['val'])} disponibles")
    if len(positive_test) < 1000:
        availability_warnings.append(f"arma test: solo {len(positive_test)} disponibles")
    if len(negative_test) < 1000:
        availability_warnings.append(f"no_arma test: solo {len(negative_test)} disponibles")

    copy_stats: Dict[str, Dict[str, int]] = {}
    copy_stats["train/arma"] = copy_images(
        positive_split["train"], dataset_root / "train" / "arma", prefix="arma_train"
    )
    copy_stats["val/arma"] = copy_images(
        positive_split["val"], dataset_root / "val" / "arma", prefix="arma_val"
    )
    copy_stats["test/arma"] = copy_images(
        positive_test, dataset_root / "test" / "arma", prefix="arma_test"
    )
    copy_stats["train/no_arma"] = copy_images(
        negative_split["train"], dataset_root / "train" / "no_arma", prefix="no_arma_train"
    )
    copy_stats["val/no_arma"] = copy_images(
        negative_split["val"], dataset_root / "val" / "no_arma", prefix="no_arma_val"
    )
    copy_stats["test/no_arma"] = copy_images(
        negative_test, dataset_root / "test" / "no_arma", prefix="no_arma_test"
    )

    summary = {
        "train/arma": count_jpgs(dataset_root / "train" / "arma"),
        "val/arma": count_jpgs(dataset_root / "val" / "arma"),
        "test/arma": count_jpgs(dataset_root / "test" / "arma"),
        "train/no_arma": count_jpgs(dataset_root / "train" / "no_arma"),
        "val/no_arma": count_jpgs(dataset_root / "val" / "no_arma"),
        "test/no_arma": count_jpgs(dataset_root / "test" / "no_arma"),
    }

    print("\nDataset NASNet creado en:")
    print(dataset_root)

    if availability_warnings:
        print("\nAvisos de disponibilidad:")
        for warning in availability_warnings:
            print(f"- {warning}")

    print("\nResumen de copiado:")
    for key, stats in copy_stats.items():
        print(f"- {key}: requested={stats['requested']}, copied={stats['copied']}, invalid={stats['invalid']}")

    print("\nConteo final por carpeta:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
