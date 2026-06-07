from __future__ import annotations

import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from PIL import Image
from tqdm import tqdm


SEED = 42
IMAGE_SUFFIX = ".jpg"
CHUNK_SIZE = 1024 * 1024
TARGET_TRAIN_ARMA = 45000
TARGET_TRAIN_NO_ARMA = 50000


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_root() -> Path:
    return project_root() / "workspace_modelos" / "dataset_clasificacion_nasnet"


def list_jpgs(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob(f"*{IMAGE_SUFFIX}") if path.is_file())


def count_jpgs(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == IMAGE_SUFFIX)


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_current_counts(root: Path) -> Dict[str, int]:
    mapping = {}
    for split in ("train", "val", "test"):
        for label in ("arma", "no_arma"):
            directory = root / split / label
            mapping[f"{split}/{label}"] = count_jpgs(directory)
    return mapping


def build_used_hashes(root: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    used_hashes: Set[str] = set()
    existing_names: Dict[str, Set[str]] = defaultdict(set)

    all_dirs = [
        root / "train" / "arma",
        root / "train" / "no_arma",
        root / "val" / "arma",
        root / "val" / "no_arma",
        root / "test" / "arma",
        root / "test" / "no_arma",
    ]

    all_files: List[Tuple[str, Path]] = []
    for directory in all_dirs:
        for file_path in list_jpgs(directory):
            all_files.append((str(directory), file_path))

    for directory_key, file_path in tqdm(all_files, desc="Hashing dataset actual", unit="img"):
        file_hash = compute_sha256(file_path)
        used_hashes.add(file_hash)
        existing_names[directory_key].add(file_path.name)

    return used_hashes, existing_names


def safe_destination_name(destination_dir: Path, source_name: str, existing_names: Set[str], stats: Dict[str, int]) -> str:
    if source_name not in existing_names:
        existing_names.add(source_name)
        return source_name

    stem = Path(source_name).stem
    suffix = Path(source_name).suffix.lower()
    stats["renamed_due_name_collision"] += 1
    for index in range(1, 1000000):
        candidate = f"{stem}__fase2_{index:05d}{suffix}"
        if candidate not in existing_names and not (destination_dir / candidate).exists():
            existing_names.add(candidate)
            return candidate
    raise RuntimeError(f"No se pudo generar un nombre unico para {source_name}")


def materialize_image(src: Path, dst: Path, stats: Dict[str, int]) -> None:
    try:
        os.link(src, dst)
        stats["added_hardlink"] += 1
    except OSError:
        shutil.copy2(src, dst)
        stats["added_copy"] += 1


def add_images_to_train(
    candidates: Iterable[Path],
    destination_dir: Path,
    target_count: int,
    used_hashes: Set[str],
    existing_names: Set[str],
    stats: Dict[str, int],
    label_key: str,
) -> None:
    current_count = count_jpgs(destination_dir)
    if current_count >= target_count:
        return

    for source_path in tqdm(list(candidates), desc=f"Aumentando {label_key}", unit="img"):
        if current_count >= target_count:
            break

        if not verify_image(source_path):
            stats["skipped_corrupt"] += 1
            continue

        source_hash = compute_sha256(source_path)
        if source_hash in used_hashes:
            stats["skipped_duplicate_hash"] += 1
            continue

        destination_name = safe_destination_name(destination_dir, source_path.name, existing_names, stats)
        destination_path = destination_dir / destination_name
        materialize_image(source_path, destination_path, stats)
        used_hashes.add(source_hash)
        current_count += 1
        stats[f"added_{label_key}"] += 1


def source_positive_candidates(root: Path) -> List[Path]:
    return list_jpgs(root / "classifier" / "gun" / "train")


def source_negative_candidates(root: Path) -> List[Path]:
    return list_jpgs(root / "classifier" / "other" / "other" / "train")


def main() -> None:
    root = project_root()
    data_root = dataset_root()
    rng = random.Random(SEED)

    required_dirs = [
        data_root / "train" / "arma",
        data_root / "train" / "no_arma",
        data_root / "val" / "arma",
        data_root / "val" / "no_arma",
        data_root / "test" / "arma",
        data_root / "test" / "no_arma",
    ]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"No existe la carpeta esperada: {directory}")

    initial_counts = inspect_current_counts(data_root)
    used_hashes, existing_names = build_used_hashes(data_root)

    positive_candidates = source_positive_candidates(root)
    negative_candidates = source_negative_candidates(root)
    rng.shuffle(positive_candidates)
    rng.shuffle(negative_candidates)

    stats: Dict[str, int] = defaultdict(int)

    add_images_to_train(
        candidates=positive_candidates,
        destination_dir=data_root / "train" / "arma",
        target_count=TARGET_TRAIN_ARMA,
        used_hashes=used_hashes,
        existing_names=existing_names[str(data_root / "train" / "arma")],
        stats=stats,
        label_key="arma",
    )

    add_images_to_train(
        candidates=negative_candidates,
        destination_dir=data_root / "train" / "no_arma",
        target_count=TARGET_TRAIN_NO_ARMA,
        used_hashes=used_hashes,
        existing_names=existing_names[str(data_root / "train" / "no_arma")],
        stats=stats,
        label_key="no_arma",
    )

    final_counts = inspect_current_counts(data_root)
    reached_arma = final_counts["train/arma"] >= TARGET_TRAIN_ARMA
    reached_no_arma = final_counts["train/no_arma"] >= TARGET_TRAIN_NO_ARMA

    print("\nConteo inicial:")
    for key, value in initial_counts.items():
        print(f"- {key}: {value}")

    print("\nAgregado Fase 2:")
    print(f"- imagenes agregadas train/arma: {stats['added_arma']}")
    print(f"- imagenes agregadas train/no_arma: {stats['added_no_arma']}")

    print("\nConteo final:")
    for key, value in final_counts.items():
        print(f"- {key}: {value}")

    print("\nResumen tecnico:")
    print(f"- agregadas por hardlink: {stats['added_hardlink']}")
    print(f"- agregadas por copia: {stats['added_copy']}")
    print(f"- omitidas por hash duplicado: {stats['skipped_duplicate_hash']}")
    print(f"- renombradas por colision de nombre: {stats['renamed_due_name_collision']}")
    print(f"- omitidas por error/corrupcion: {stats['skipped_corrupt']}")
    print(f"- objetivo train/arma alcanzado: {reached_arma}")
    print(f"- objetivo train/no_arma alcanzado: {reached_no_arma}")


if __name__ == "__main__":
    main()
