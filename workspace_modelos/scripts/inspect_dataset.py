from __future__ import annotations

import argparse
from pathlib import Path

from utils_paths import dataset_source_paths, resolve_project_root
from utils_voc import list_images, parse_voc_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspecciona el dataset original en modo solo lectura.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-xml", type=int, default=200, help="Cantidad maxima de XML a validar en detalle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    paths = dataset_source_paths(project_root)

    print("Rutas fuente:")
    for key, value in paths.items():
        print(f"  - {key}: {value}")

    print("\nConteos:")
    for key, value in paths.items():
        images = len(list_images(value))
        xmls = len(list(value.rglob("*.xml"))) if value.exists() else 0
        print(f"  - {key}: imagenes={images}, xml={xmls}")

    xml_paths = sorted(paths["gun_train_xml"].rglob("*.xml"))[: args.sample_xml]
    invalid = 0
    empty = 0
    for xml_path in xml_paths:
        try:
            annotation = parse_voc_xml(xml_path)
            if not annotation.objects:
                empty += 1
        except Exception:
            invalid += 1

    print("\nValidacion de muestra:")
    print(f"  - XML revisados: {len(xml_paths)}")
    print(f"  - XML invalidos: {invalid}")
    print(f"  - XML sin objetos: {empty}")


if __name__ == "__main__":
    main()
