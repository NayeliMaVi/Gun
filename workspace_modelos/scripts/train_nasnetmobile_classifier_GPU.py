from __future__ import annotations

import csv
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
INITIAL_EPOCHS = 10
USE_MIXED_PRECISION = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_directories() -> dict[str, Path]:
    root = project_root()
    paths = {
        "dataset_root": root / "workspace_modelos" / "dataset_clasificacion_nasnet",
        "train_dir": root / "workspace_modelos" / "dataset_clasificacion_nasnet" / "train",
        "val_dir": root / "workspace_modelos" / "dataset_clasificacion_nasnet" / "val",
        "test_dir": root / "workspace_modelos" / "dataset_clasificacion_nasnet" / "test",
        "models_dir": root / "workspace_modelos" / "models",
        "reports_dir": root / "workspace_modelos" / "reports",
    }
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def configure_gpu() -> list[tf.config.PhysicalDevice]:
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detectadas:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if USE_MIXED_PRECISION and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision activada: mixed_float16")
    else:
        print("Mixed precision desactivada")
    return gpus


def build_datasets(paths: dict[str, Path]) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = image_dataset_from_directory(
        paths["train_dir"],
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = image_dataset_from_directory(
        paths["val_dir"],
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = image_dataset_from_directory(
        paths["test_dir"],
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_augmentation() -> tf.keras.Sequential:
    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ]
    if hasattr(layers, "RandomContrast"):
        augmentation_layers.append(layers.RandomContrast(0.10))
    return tf.keras.Sequential(augmentation_layers, name="train_augmentation")


def build_model() -> tf.keras.Model:
    data_augmentation = build_augmentation()
    base_model = NASNetMobile(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_input, name="nasnet_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="binary_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="nasnetmobile_weapon_validator_gpu")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_callbacks(paths: dict[str, Path]) -> list[tf.keras.callbacks.Callback]:
    best_model_path = paths["models_dir"] / "nasnetmobile_weapon_validator_best_GPU.keras"
    return [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def save_history(history: tf.keras.callbacks.History, output_csv: Path) -> None:
    rows = history.history
    fieldnames = list(rows.keys())
    total_rows = len(rows[fieldnames[0]]) if fieldnames else 0

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(total_rows):
            writer.writerow({key: rows[key][row_index] for key in fieldnames})


def save_test_metrics(metrics: dict[str, float], output_txt: Path, class_names: list[str], gpus: list) -> None:
    with output_txt.open("w", encoding="utf-8") as handle:
        handle.write("NasNetMobile GPU test metrics\n")
        handle.write(f"class_names: {class_names}\n")
        handle.write(f"gpus: {gpus}\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")


def main() -> None:
    paths = ensure_directories()
    gpus = configure_gpu()
    if not gpus:
        raise RuntimeError("TensorFlow no detecto GPU. Revisa el entorno Debian/WSL2.")

    train_ds, val_ds, test_ds, class_names = build_datasets(paths)

    print("class_names:", class_names)
    print("Orden esperado para binario:")
    print("  - 0.0 ->", class_names[0])
    print("  - 1.0 ->", class_names[1])

    model = build_model()
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=build_callbacks(paths),
        verbose=1,
    )

    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    print("Test metrics:", test_metrics)

    final_model_path = paths["models_dir"] / "nasnetmobile_weapon_validator_final_GPU.keras"
    history_csv_path = paths["reports_dir"] / "nasnetmobile_training_history_GPU.csv"
    metrics_txt_path = paths["reports_dir"] / "nasnetmobile_test_metrics_GPU.txt"

    model.save(final_model_path)
    save_history(history, history_csv_path)
    save_test_metrics(test_metrics, metrics_txt_path, class_names, gpus)

    print("Modelo final guardado en:", final_model_path)
    print("Historial guardado en:", history_csv_path)
    print("Metricas guardadas en:", metrics_txt_path)


if __name__ == "__main__":
    main()
