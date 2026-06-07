from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.nasnet import preprocess_input


YOLO_CONF = 0.25
YOLO_IMGSZ = 960
NASNET_THRESHOLD = 0.5
DEBUG = True
IOU_DUPLICATE_THRESHOLD = 0.5


@dataclass
class DetectionCandidate:
    box_xyxy: Tuple[int, int, int, int]
    yolo_score: float
    arma_score: float
    raw_score: float
    predicted_label: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_class_names(train_dir: Path) -> List[str]:
    class_names = sorted([path.name for path in train_dir.iterdir() if path.is_dir()])
    if len(class_names) != 2:
        raise ValueError(f"Se esperaban 2 clases binarias y se encontraron: {class_names}")
    return class_names


def get_arma_score(raw_score: float, class_names: Sequence[str]) -> Tuple[str, float]:
    if list(class_names) == ["arma", "no_arma"]:
        arma_score = 1.0 - raw_score
    elif list(class_names) == ["no_arma", "arma"]:
        arma_score = raw_score
    else:
        raise ValueError(f"Orden de clases no soportado para binario: {class_names}")

    predicted_label = "arma" if arma_score >= NASNET_THRESHOLD else "no_arma"
    return predicted_label, arma_score


def predict_nasnet_crop(
    crop_bgr: np.ndarray,
    nasnet_model: tf.keras.Model,
    class_names: Sequence[str],
) -> Tuple[str, float, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    raw_score = float(nasnet_model.predict(batch, verbose=0)[0][0])
    predicted_label, arma_score = get_arma_score(raw_score, class_names)
    return predicted_label, arma_score, raw_score


def clip_box_to_frame(box_xyxy: Sequence[float], frame_shape: Sequence[int]) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def extract_crop(frame_bgr: np.ndarray, box_xyxy: Sequence[float]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = clip_box_to_frame(box_xyxy, frame_bgr.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def compute_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def suppress_duplicates(candidates: Sequence[DetectionCandidate], iou_threshold: float) -> List[DetectionCandidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (item.arma_score, item.yolo_score),
        reverse=True,
    )
    selected: List[DetectionCandidate] = []
    for candidate in sorted_candidates:
        keep = True
        for chosen in selected:
            if compute_iou(candidate.box_xyxy, chosen.box_xyxy) > iou_threshold:
                keep = False
                break
        if keep:
            selected.append(candidate)
    return selected


def draw_detection(
    frame_bgr: np.ndarray,
    box_xyxy: Sequence[int],
    text: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame_bgr,
        text,
        (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    root = project_root()

    yolo_weapon_path = root / "workspace_modelos" / "runs" / "yolo" / "weapon_yolo_full" / "weights" / "best.pt"
    nasnet_model_path = root / "workspace_modelos" / "models" / "nasnetmobile_weapon_validator_final_GPU.keras"
    class_train_dir = root / "workspace_modelos" / "dataset_clasificacion_nasnet" / "train"
    input_video_path = root / "video" / "prueba3.mp4"
    output_video_path = root / "workspace_modelos" / "reports" / "video_yolo_nasnet" / "prueba_yolo_nasnet3.mp4"

    ensure_output_dir(output_video_path)

    if not yolo_weapon_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo YOLO arma: {yolo_weapon_path}")
    if not nasnet_model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo NASNetMobile: {nasnet_model_path}")
    if not input_video_path.exists():
        raise FileNotFoundError(f"No se encontro el video de entrada: {input_video_path}")
    if not class_train_dir.exists():
        raise FileNotFoundError(f"No se encontro el train de clasificacion: {class_train_dir}")

    class_names = load_class_names(class_train_dir)
    print("class_names detectadas:", class_names)

    yolo_weapon = YOLO(str(yolo_weapon_path))
    nasnet_model = tf.keras.models.load_model(
        str(nasnet_model_path),
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {output_video_path}")

    stats = {
        "frames_procesados": 0,
        "detecciones_yolo_totales": 0,
        "detecciones_confirmadas_nasnet": 0,
        "detecciones_rechazadas_nasnet": 0,
    }

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        stats["frames_procesados"] += 1

        weapon_results = yolo_weapon.predict(
            source=frame_bgr,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        result_weapon = weapon_results[0]

        confirmed_candidates: List[DetectionCandidate] = []
        rejected_candidates: List[DetectionCandidate] = []

        if result_weapon.boxes is not None:
            for box in result_weapon.boxes:
                stats["detecciones_yolo_totales"] += 1
                box_xyxy = clip_box_to_frame(box.xyxy[0].cpu().numpy().tolist(), frame_bgr.shape)
                yolo_score = float(box.conf[0].item())

                crop = extract_crop(frame_bgr, box_xyxy)
                if crop is None or crop.size == 0:
                    stats["detecciones_rechazadas_nasnet"] += 1
                    continue

                predicted_label, arma_score, raw_score = predict_nasnet_crop(crop, nasnet_model, class_names)
                candidate = DetectionCandidate(
                    box_xyxy=box_xyxy,
                    yolo_score=yolo_score,
                    arma_score=arma_score,
                    raw_score=raw_score,
                    predicted_label=predicted_label,
                )

                if predicted_label == "arma" and arma_score >= NASNET_THRESHOLD:
                    confirmed_candidates.append(candidate)
                else:
                    stats["detecciones_rechazadas_nasnet"] += 1
                    rejected_candidates.append(candidate)

        confirmed_candidates = suppress_duplicates(confirmed_candidates, IOU_DUPLICATE_THRESHOLD)
        stats["detecciones_confirmadas_nasnet"] += len(confirmed_candidates)

        for candidate in confirmed_candidates:
            draw_detection(
                frame_bgr,
                candidate.box_xyxy,
                f"ARMA CONFIRMADA | yolo={candidate.yolo_score:.2f} | arma={candidate.arma_score:.2f}",
                color=(0, 0, 255),
                thickness=2,
            )

        if DEBUG:
            rejected_candidates = suppress_duplicates(rejected_candidates, IOU_DUPLICATE_THRESHOLD)
            for candidate in rejected_candidates:
                draw_detection(
                    frame_bgr,
                    candidate.box_xyxy,
                    f"rechazado | yolo={candidate.yolo_score:.2f} | arma={candidate.arma_score:.2f} | raw={candidate.raw_score:.2f}",
                    color=(0, 255, 255),
                    thickness=2,
                )

        writer.write(frame_bgr)

    cap.release()
    writer.release()

    print("\nResumen final:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    print(f"- video_salida: {output_video_path}")


if __name__ == "__main__":
    main()
