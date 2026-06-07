from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils_paths import resolve_project_root, workspace_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO usando el dataset externo de workspace_modelos.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", type=str, default="weapon_yolo")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    paths = workspace_paths(project_root)
    data_yaml = paths["yolo_dataset"] / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe {data_yaml}. Ejecuta primero prepare_yolo_dataset.py")

    if args.resume:
        last_weights = paths["runs_yolo"] / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"No existe checkpoint para reanudar: {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(paths["runs_yolo"]),
        name=args.name,
        exist_ok=True,
        save=True,
    )


if __name__ == "__main__":
    main()
