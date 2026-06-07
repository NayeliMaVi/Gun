from __future__ import annotations

from pathlib import Path
from typing import Dict


def resolve_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "classifier").exists() and (candidate / "detector").exists():
            return candidate
    raise FileNotFoundError("No se encontro la raiz del proyecto con classifier/ y detector/.")


def workspace_paths(project_root: Path) -> Dict[str, Path]:
    workspace_root = project_root / "workspace_modelos"
    return {
        "workspace_root": workspace_root,
        "config": workspace_root / "config",
        "scripts": workspace_root / "scripts",
        "reports": workspace_root / "reports",
        "logs": workspace_root / "logs",
        "datasets": workspace_root / "datasets",
        "yolo_dataset": workspace_root / "datasets" / "yolo_gun",
        "runs_yolo": workspace_root / "runs" / "yolo",
    }


def dataset_source_paths(project_root: Path) -> Dict[str, Path]:
    detector_root = project_root / "detector"
    return {
        "gun_train_images": detector_root / "gun" / "Train" / "JPEGImages",
        "gun_train_xml": detector_root / "gun" / "Train" / "Annotations",
        "gun_test_images": detector_root / "gun" / "Test" / "JPEGImages",
        "gun_test_xml": detector_root / "gun" / "Test" / "Annotations",
        "other_images": detector_root / "other",
    }
