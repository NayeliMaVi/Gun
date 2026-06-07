from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import deque

# ============================================================
# RUTAS
# ============================================================
POSE_MODEL_PATH   = "yolov8n-pose.pt"
WEAPON_MODEL_PATH = "workspace_modelos/runs/yolo/weapon_yolo_full/weights/best.pt"
VIDEO_PATH        = "video/prueba.mp4"
OUTPUT_DIR        = "workspace_modelos/reports/video_pose_arma"
OUTPUT_VIDEO      = os.path.join(OUTPUT_DIR, "prueba_pose_arma_video.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURACIÓN — ajusta estos valores sin tocar el resto
# ============================================================
POSE_CONF    = 0.35   # confianza mínima para detectar la pose de una persona
WEAPON_CONF  = 0.30   # confianza mínima para aceptar una detección de arma
POSE_IMGSZ   = 640    # resolución de inferencia para el modelo de pose
WEAPON_IMGSZ = 640    # resolución de inferencia para el modelo de armas

KP_CONF_MIN  = 0.30   # confianza mínima de un keypoint para usarlo (filtra puntos inseguros)

# Padding alrededor de muñecas/codos — relativo al "span" de la mano, no absoluto
# Se multiplica por la distancia hombro-codo para escalar con la persona en cámara
WRIST_PAD_FACTOR  = 0.6   # zona pequeña centrada en muñeca (zona primaria)
ELBOW_PAD_FACTOR  = 0.4   # zona secundaria que incluye antebrazo

# Persistencia temporal: alerta SOLO si arma aparece en >= ALERT_MIN_HITS de los últimos ALERT_WINDOW frames
ALERT_WINDOW   = 5
ALERT_MIN_HITS = 2

# ============================================================
# ÍNDICES COCO KEYPOINTS
# ============================================================
# 5=hombro_izq  6=hombro_der
# 7=codo_izq    8=codo_der
# 9=muñeca_izq  10=muñeca_der
LEFT_ARM  = (5, 7, 9)   # hombro, codo, muñeca izquierda
RIGHT_ARM = (6, 8, 10)  # hombro, codo, muñeca derecha

# ============================================================
# HELPERS
# ============================================================

def get_keypoint(kps, kps_conf, idx):
    """
    Devuelve (x, y, conf) de un keypoint si tiene confianza suficiente,
    o None si el punto no es fiable.
    """
    x, y = kps[idx]
    conf = float(kps_conf[idx]) if kps_conf is not None else 1.0
    if x <= 0 or y <= 0 or conf < KP_CONF_MIN:
        return None
    return int(x), int(y), conf


def arm_scale(kps, kps_conf, shoulder_idx, elbow_idx):
    """
    Estima la longitud del segmento hombro→codo como referencia de escala.
    Retorna la distancia en píxeles, o None si no hay ambos puntos.
    """
    s = get_keypoint(kps, kps_conf, shoulder_idx)
    e = get_keypoint(kps, kps_conf, elbow_idx)
    if s is None or e is None:
        return None
    return np.hypot(s[0] - e[0], s[1] - e[1])


def build_zone(center_x, center_y, pad, frame_w, frame_h):
    """Crea un rectángulo cuadrado centrado en (center_x, center_y) con radio `pad`."""
    x1 = max(0, center_x - pad)
    y1 = max(0, center_y - pad)
    x2 = min(frame_w, center_x + pad)
    y2 = min(frame_h, center_y + pad)
    return x1, y1, x2, y2


def detect_weapon_in_zone(frame, zone, weapon_model):
    """
    Recorta `zone` del frame, corre el modelo de armas y devuelve la caja
    con mayor confianza de clase 'gun', o None si no hay detección.
    Retorna dict con keys: wx1,wy1,wx2,wy2 (coordenadas globales), conf.
    """
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    results = weapon_model.predict(
        source=crop,
        imgsz=WEAPON_IMGSZ,
        conf=WEAPON_CONF,
        max_det=5,
        verbose=False
    )

    best = None
    for box in results[0].boxes:
        cls_name = weapon_model.names[int(box.cls[0])]
        if cls_name != "gun":
            continue
        conf = float(box.conf[0])
        if best is None or conf > best["conf"]:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            best = {
                "wx1": x1 + bx1, "wy1": y1 + by1,
                "wx2": x1 + bx2, "wy2": y1 + by2,
                "conf": conf
            }
    return best


def draw_zone(frame, zone, color=(255, 140, 0), label=""):
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    if label:
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_weapon(frame, det, alert_active):
    color = (0, 0, 255) if alert_active else (0, 165, 255)
    cv2.rectangle(frame, (det["wx1"], det["wy1"]), (det["wx2"], det["wy2"]), color, 3)
    label = f"{'[ALERTA] ' if alert_active else ''}ARMA {det['conf']:.2f}"
    cv2.putText(frame, label, (det["wx1"], max(det["wy1"] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


# ============================================================
# MAIN
# ============================================================

def process_video():
    pose_model   = YOLO(POSE_MODEL_PATH)
    weapon_model = YOLO(WEAPON_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Ventana deslizante de persistencia temporal
    detection_history = deque(maxlen=ALERT_WINDOW)

    frame_number      = 0
    frames_con_arma   = 0
    frames_con_alerta = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        arma_en_frame = False

        # --------------------------------------------------
        # 1. POSE ESTIMATION
        # --------------------------------------------------
        pose_results = pose_model.predict(
            source=frame, imgsz=POSE_IMGSZ, conf=POSE_CONF, verbose=False
        )
        pose_result = pose_results[0]

        if pose_result.keypoints is None:
            detection_history.append(False)
            out.write(frame)
            continue

        kps_xy   = pose_result.keypoints.xy.cpu().numpy()
        # kps_conf puede estar disponible o no según el modelo
        kps_conf_all = (pose_result.keypoints.conf.cpu().numpy()
                        if pose_result.keypoints.conf is not None else None)

        best_detection = None   # guardamos la detección más confiable del frame

        # --------------------------------------------------
        # 2. POR CADA PERSONA DETECTADA
        # --------------------------------------------------
        for person_idx, kps in enumerate(kps_xy):
            kps_conf = kps_conf_all[person_idx] if kps_conf_all is not None else None

            for side_name, (sh_idx, el_idx, wr_idx) in [
                ("IZQ", LEFT_ARM), ("DER", RIGHT_ARM)
            ]:
                wrist  = get_keypoint(kps, kps_conf, wr_idx)
                elbow  = get_keypoint(kps, kps_conf, el_idx)

                # Sin muñeca no hay zona útil
                if wrist is None:
                    continue

                scale = arm_scale(kps, kps_conf, sh_idx, el_idx)
                if scale is None or scale < 10:
                    scale = 60   # fallback razonable si no se ve el hombro

                # ── Zona primaria: centrada en muñeca ──────────────────
                # ── Zona extendida desde codo hacia muñeca/mano ─────────────
                pad_wrist = int(scale * 0.9)

                if elbow is not None:
                    # Vector desde codo hacia muñeca
                    vx = wrist[0] - elbow[0]
                    vy = wrist[1] - elbow[1]

                    # Extendemos la zona más allá de la muñeca,
                    # porque el arma suele estar después de la mano
                    extended_x = int(wrist[0] + vx * 0.9)
                    extended_y = int(wrist[1] + vy * 0.9)

                    x_min = min(elbow[0], wrist[0], extended_x) - pad_wrist
                    y_min = min(elbow[1], wrist[1], extended_y) - pad_wrist
                    x_max = max(elbow[0], wrist[0], extended_x) + pad_wrist
                    y_max = max(elbow[1], wrist[1], extended_y) + pad_wrist

                    zone_wrist = (
                        max(0, x_min),
                        max(0, y_min),
                        min(width, x_max),
                        min(height, y_max)
                    )
                else:
                    zone_wrist = build_zone(wrist[0], wrist[1], pad_wrist, width, height)

                draw_zone(
                    frame,
                    zone_wrist,
                    color=(255, 200, 0),
                    label=f"mano extendida {side_name}"
                )

                # ── Zona secundaria: centrada en codo/antebrazo ────────
                if elbow is not None:
                    # Punto medio entre codo y muñeca = antebrazo
                    mid_x = (elbow[0] + wrist[0]) // 2
                    mid_y = (elbow[1] + wrist[1]) // 2
                    pad_elbow = int(scale * ELBOW_PAD_FACTOR)
                    zone_elbow = build_zone(mid_x, mid_y, pad_elbow, width, height)
                    draw_zone(frame, zone_elbow, color=(200, 180, 0),
                              label=f"antebrazo {side_name}")
                else:
                    zone_elbow = None

                # Dibujar keypoints usados
                cv2.circle(frame, (wrist[0], wrist[1]), 5, (0, 255, 255), -1)
                if elbow:
                    cv2.circle(frame, (elbow[0], elbow[1]), 5, (0, 200, 200), -1)

                # ── Detección de arma en zona primaria ────────────────
                det = detect_weapon_in_zone(frame, zone_wrist, weapon_model)

                # Si no encontró en la muñeca, intenta en el antebrazo
                if det is None and zone_elbow is not None:
                    det = detect_weapon_in_zone(frame, zone_elbow, weapon_model)

                # Nos quedamos con la detección más confiable del frame
                if det is not None:
                    if best_detection is None or det["conf"] > best_detection["conf"]:
                        best_detection = det

        # --------------------------------------------------
        # 3. PERSISTENCIA TEMPORAL
        # --------------------------------------------------
        if best_detection is not None:
            arma_en_frame = True
            frames_con_arma += 1

        detection_history.append(arma_en_frame)

        # Alerta SOLO si en la ventana hay suficientes hits consecutivos
        hits = sum(detection_history)
        alert_active = hits >= ALERT_MIN_HITS

        if alert_active:
            frames_con_alerta += 1

        # Dibujamos UNA SOLA caja (la más confiable del frame)
        if best_detection is not None:
            draw_weapon(frame, best_detection, alert_active)

        # HUD de estado
        status_color = (0, 0, 255) if alert_active else (0, 200, 0)
        status_text  = f"ALERTA: ARMA ({hits}/{ALERT_WINDOW})" if alert_active \
                       else f"Sin arma ({hits}/{ALERT_WINDOW})"
        cv2.putText(frame, status_text, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Frame {frame_number}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        out.write(frame)

    cap.release()
    out.release()

    print("=" * 50)
    print("Proceso terminado.")
    print(f"Video guardado en:          {OUTPUT_VIDEO}")
    print(f"Frames procesados:          {frame_number}")
    print(f"Frames con arma detectada:  {frames_con_arma}")
    print(f"Frames con alerta activa:   {frames_con_alerta}")
    print("=" * 50)


if __name__ == "__main__":
    process_video()

# from ultralytics import YOLO
# import cv2
# import os

# # ==========================
# # RUTAS
# # ==========================
# MODEL_PATH = "workspace_modelos/runs/yolo/weapon_yolo_full/weights/best.pt"
# VIDEO_PATH = "video/prueba2.mp4"

# OUTPUT_DIR = "workspace_modelos/reports/video_arma_encuadrada"
# OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "prueba_arma_encuadrada2.mp4")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================
# # CONFIGURACIÓN
# # ==========================
# CONF_THRESHOLD = 0.25

# # ==========================
# # CARGAR MODELO
# # ==========================
# model = YOLO(MODEL_PATH)

# # ==========================
# # ABRIR VIDEO
# # ==========================
# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():
#     print("No se pudo abrir el video.")
#     exit()

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# frame_number = 0
# frames_con_arma = 0

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame_number += 1

#     results = model.predict(
#         source=frame,
#         conf=CONF_THRESHOLD,
#         verbose=False
#     )

#     result = results[0]

#     arma_detectada = False

#     if result.boxes is not None and len(result.boxes) > 0:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             class_name = model.names[cls_id]

#             if class_name == "gun":
#                 arma_detectada = True

#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

#                 x1 = max(0, x1)
#                 y1 = max(0, y1)
#                 x2 = min(width, x2)
#                 y2 = min(height, y2)

#                 # Caja del arma
#                 cv2.rectangle(
#                     frame,
#                     (x1, y1),
#                     (x2, y2),
#                     (0, 255, 0),
#                     3
#                 )

#                 # Texto encima de la caja
#                 label = f"ARMA DETECTADA {conf:.2f}"

#                 cv2.putText(
#                     frame,
#                     label,
#                     (x1, max(y1 - 10, 25)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )

#     if arma_detectada:
#         frames_con_arma += 1

#         # Alerta superior en el video
#         cv2.rectangle(
#             frame,
#             (10, 10),
#             (360, 55),
#             (0, 0, 255),
#             -1
#         )

#         cv2.putText(
#             frame,
#             "ALERTA: ARMA DETECTADA",
#             (20, 42),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 255, 255),
#             2
#         )

#     out.write(frame)

# cap.release()
# out.release()

# print("Proceso terminado.")
# print(f"Video generado en: {OUTPUT_VIDEO}")
# print(f"Frames procesados: {frame_number}")
# print(f"Frames con arma detectada: {frames_con_arma}")