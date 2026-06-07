from ultralytics import YOLO
import cv2
import os
import numpy as np


# ============================================================
# RUTAS
# ============================================================
POSE_MODEL_PATH = "yolov8n-pose.pt"
WEAPON_MODEL_PATH = "workspace_modelos/runs/yolo/weapon_yolo_full/weights/best.pt"

# ============================================================
# CONFIGURACIÓN
# ============================================================
POSE_CONF = 0.35
WEAPON_CONF = 0.25

POSE_IMGSZ = 640
WEAPON_IMGSZ = 960

KP_CONF_MIN = 0.20

# Extensión de zona de mano/brazo
WRIST_PAD_FACTOR = 1.2
ELBOW_PAD_FACTOR = 0.7



# Cámara: 0 suele ser webcam principal
CAMERA_INDEX = 0

# ============================================================
# ÍNDICES COCO
# ============================================================
LEFT_ARM = (5, 7, 9)    # hombro, codo, muñeca izquierda
RIGHT_ARM = (6, 8, 10)  # hombro, codo, muñeca derecha


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def get_keypoint(kps, kps_conf, idx):
    x, y = kps[idx]
    conf = float(kps_conf[idx]) if kps_conf is not None else 1.0

    if x <= 0 or y <= 0 or conf < KP_CONF_MIN:
        return None

    return int(x), int(y), conf


def arm_scale(kps, kps_conf, shoulder_idx, elbow_idx):
    shoulder = get_keypoint(kps, kps_conf, shoulder_idx)
    elbow = get_keypoint(kps, kps_conf, elbow_idx)

    if shoulder is None or elbow is None:
        return None

    return np.hypot(shoulder[0] - elbow[0], shoulder[1] - elbow[1])


def build_zone(center_x, center_y, pad, frame_w, frame_h):
    x1 = max(0, center_x - pad)
    y1 = max(0, center_y - pad)
    x2 = min(frame_w, center_x + pad)
    y2 = min(frame_h, center_y + pad)

    return x1, y1, x2, y2


def build_extended_hand_zone(elbow, wrist, scale, frame_w, frame_h):
    """
    Crea una zona que va desde el codo hacia la muñeca
    y se extiende más allá de la mano.
    """
    pad = int(scale * WRIST_PAD_FACTOR)

    if elbow is None:
        return build_zone(wrist[0], wrist[1], pad, frame_w, frame_h)

    vx = wrist[0] - elbow[0]
    vy = wrist[1] - elbow[1]

    extended_x = int(wrist[0] + vx * 0.9)
    extended_y = int(wrist[1] + vy * 0.9)

    x_min = min(elbow[0], wrist[0], extended_x) - pad
    y_min = min(elbow[1], wrist[1], extended_y) - pad
    x_max = max(elbow[0], wrist[0], extended_x) + pad
    y_max = max(elbow[1], wrist[1], extended_y) + pad

    return (
        max(0, x_min),
        max(0, y_min),
        min(frame_w, x_max),
        min(frame_h, y_max),
    )


def detect_weapon_in_zone(frame, zone, weapon_model):
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    results = weapon_model.predict(
        source=crop,
        imgsz=WEAPON_IMGSZ,
        conf=WEAPON_CONF,
        max_det=5,
        verbose=False,
    )

    best = None

    if results[0].boxes is None:
        return None

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = weapon_model.names[cls_id]
        conf = float(box.conf[0])

        if cls_name != "gun":
            continue

        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)

        det = {
            "x1": x1 + bx1,
            "y1": y1 + by1,
            "x2": x1 + bx2,
            "y2": y1 + by2,
            "conf": conf,
        }

        if best is None or conf > best["conf"]:
            best = det

    return best


def draw_zone(frame, zone, label):
    x1, y1, x2, y2 = zone

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1,
    )


def draw_weapon(frame, det, alert_active):
    color = (0, 0, 255) if alert_active else (0, 165, 255)

    cv2.rectangle(
        frame,
        (det["x1"], det["y1"]),
        (det["x2"], det["y2"]),
        color,
        3,
    )

    label = f"ARMA {det['conf']:.2f}"

    cv2.putText(
        frame,
        label,
        (det["x1"], max(det["y1"] - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
pose_model = YOLO(POSE_MODEL_PATH)
weapon_model = YOLO(WEAPON_MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()


frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    frame_number += 1
    height, width = frame.shape[:2]

    best_detection = None
    arma_en_frame = False

    # ========================================================
    # 1. POSE ESTIMATION
    # ========================================================
    pose_results = pose_model.predict(
        source=frame,
        imgsz=POSE_IMGSZ,
        conf=POSE_CONF,
        verbose=False,
    )

    pose_result = pose_results[0]

    if pose_result.keypoints is not None:
        kps_xy = pose_result.keypoints.xy.cpu().numpy()

        kps_conf_all = (
            pose_result.keypoints.conf.cpu().numpy()
            if pose_result.keypoints.conf is not None
            else None
        )

        for person_idx, kps in enumerate(kps_xy):
            kps_conf = (
                kps_conf_all[person_idx]
                if kps_conf_all is not None
                else None
            )

            for side_name, (shoulder_idx, elbow_idx, wrist_idx) in [
                ("IZQ", LEFT_ARM),
                ("DER", RIGHT_ARM),
            ]:
                wrist = get_keypoint(kps, kps_conf, wrist_idx)
                elbow = get_keypoint(kps, kps_conf, elbow_idx)

                if wrist is None:
                    continue

                scale = arm_scale(kps, kps_conf, shoulder_idx, elbow_idx)

                if scale is None or scale < 10:
                    scale = 60

                # Zona extendida desde brazo hacia mano
                zone = build_extended_hand_zone(
                    elbow=elbow,
                    wrist=wrist,
                    scale=scale,
                    frame_w=width,
                    frame_h=height,
                )

                draw_zone(frame, zone, f"mano extendida {side_name}")

                # Dibujar keypoints
                cv2.circle(frame, (wrist[0], wrist[1]), 5, (0, 255, 255), -1)

                if elbow is not None:
                    cv2.circle(frame, (elbow[0], elbow[1]), 5, (0, 200, 200), -1)

                # ========================================================
                # 2. DETECCIÓN DE ARMA EN ESA ZONA
                # ========================================================
                det = detect_weapon_in_zone(frame, zone, weapon_model)

                if det is not None:
                    if best_detection is None or det["conf"] > best_detection["conf"]:
                        best_detection = det

    # ========================================================
    # 3. PERSISTENCIA TEMPORAL
    # ========================================================
    alert_active = best_detection is not None

    if best_detection is not None:
        draw_weapon(frame, best_detection, alert_active)

    # ========================================================
    # 4. HUD
    # ========================================================
    if alert_active:
        status_text = "ALERTA: ARMA"
        status_color = (0, 0, 255)
    else:
        status_text = "Sin arma"
        status_color = (0, 255, 0)

    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3,
    )

    cv2.putText(
        frame,
        f"Frame {frame_number}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )

    cv2.imshow("Camara - Pose + Deteccion de arma", frame)

    # Presionar Q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()