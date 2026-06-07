from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_CLASS_ID = 0
YOLO_CLASS_NAME = "gun"


@dataclass
class VocObject:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: List[VocObject]


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_voc_xml(xml_path: Path) -> VocAnnotation:
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or xml_path.stem).strip()
    size = root.find("size")
    if size is None:
        raise ValueError(f"XML sin bloque <size>: {xml_path}")

    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensiones invalidas en {xml_path}: {width}x{height}")

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        objects.append(
            VocObject(
                class_name=(obj.findtext("name") or "").strip().lower(),
                xmin=float(bbox.findtext("xmin", "0")),
                ymin=float(bbox.findtext("ymin", "0")),
                xmax=float(bbox.findtext("xmax", "0")),
                ymax=float(bbox.findtext("ymax", "0")),
            )
        )

    return VocAnnotation(filename=filename, width=width, height=height, objects=objects)


def validate_voc_object(obj: VocObject, width: int, height: int) -> None:
    if obj.xmax <= obj.xmin or obj.ymax <= obj.ymin:
        raise ValueError(f"Caja invalida: {(obj.xmin, obj.ymin, obj.xmax, obj.ymax)}")
    if obj.xmin < 0 or obj.ymin < 0 or obj.xmax > width or obj.ymax > height:
        raise ValueError(f"Caja fuera de rango: {(obj.xmin, obj.ymin, obj.xmax, obj.ymax)}")


def voc_to_yolo_line(obj: VocObject, width: int, height: int, class_id: int = YOLO_CLASS_ID) -> str:
    validate_voc_object(obj, width, height)
    x_center = ((obj.xmin + obj.xmax) / 2.0) / width
    y_center = ((obj.ymin + obj.ymax) / 2.0) / height
    box_width = (obj.xmax - obj.xmin) / width
    box_height = (obj.ymax - obj.ymin) / height
    values = (x_center, y_center, box_width, box_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Coordenadas YOLO fuera de [0,1]: {values}")
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def materialize_file(source: Path, destination: Path, copy_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if copy_mode == "symlink":
        try:
            os.symlink(source.resolve(), destination)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def write_label_file(label_path: Path, lines: Sequence[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def safe_stem(prefix: str, image_path: Path) -> str:
    clean_prefix = prefix.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{clean_prefix}_{image_path.stem}"
