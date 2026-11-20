#!/usr/bin/env python3
"""
DermaReach — multi-class skin disease classification for dark skin tones.

Classifies acne, eczema, and keloids using transfer learning (VGG16) with
Keras/TensorFlow. Companion code for the master's research project:

  "Classification of Selected Skin Diseases Using Artificial Intelligence
   for Dark Skin Tones" — Paul Ajadi & Nancy Woods, University of Ibadan.

Expected directory layout for --data-dir:

  data/
    acne/
    eczema/
    keloids/

Each class folder should contain JPEG images of that condition.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 64
EPOCHS = 20
VALID_EXTENSIONS = (".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DermaReach skin-disease classifier (acne / eczema / keloids)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to class folders (acne/, eczema/, keloids/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for plots and saved model weights.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained Keras model under --output-dir.",
    )
    return parser.parse_args()


def collect_image_paths(data_dir: Path) -> tuple[list[str], list[str]]:
    image_files: list[str] = []
    labels: list[str] = []

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_folders = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not class_folders:
        raise ValueError(
            f"No class subfolders found in {data_dir}. "
            "Expected folders such as acne/, eczema/, keloids/."
        )

    for class_folder in class_folders:
        for file in class_folder.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(str(file))
                labels.append(class_folder.name)

    if not image_files:
        raise ValueError(
            f"No JPEG images found under {data_dir}. "
            "Convert HEIC captures to JPG before training."
        )

    return image_files, labels


def learning_rate_schedule(epoch: int) -> float:
    initial_lr = 0.001
    decay_rate = 0.1
    if epoch < 10:
        return initial_lr
    if epoch < 20:
        return initial_lr * decay_rate
    return initial_lr * (decay_rate**2)


def build_model(num_classes: int, image_size: int) -> Model:
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_sample_batch(
    generator: ImageDataGenerator,
    class_labels: list[str],
    output_path: Path,
) -> None:
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_labels[int(np.argmax(labels[i]))])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_learning_rate(epochs: list[int], output_path: Path) -> None:
    plt.figure()
    plt.plot(epochs, [learning_rate_schedule(epoch) for epoch in epochs])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.savefig(output_path)
    plt.close()


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    plt.figure()
    for i, name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = args.image_size
    batch_size = args.batch_size

    image_files, labels = collect_image_paths(args.data_dir)
    num_classes = len(set(labels))
    print(f"Found {len(image_files)} images across {num_classes} classes.")

    train_files, validation_files, train_labels, validation_labels = train_test_split(
        image_files,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_df = pd.DataFrame({"filename": train_files, "class": train_labels})
    validation_df = pd.DataFrame(
        {"filename": validation_files, "class": validation_labels}
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.6,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="class",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=args.seed,
    )
    validation_generator = test_datagen.flow_from_dataframe(
        validation_df,
        x_col="filename",
        y_col="class",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_labels = list(train_generator.class_indices.keys())
    print("Class indices:", train_generator.class_indices)
    plot_sample_batch(
        train_generator, class_labels, output_dir / "sample_training_batch.png"
    )

    model = build_model(num_classes=num_classes, image_size=image_size)
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // batch_size),
        verbose=1,
        callbacks=[lr_scheduler],
    )

    plot_learning_rate(list(history.epoch), output_dir / "learning_rate_schedule.png")

    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f"Validation loss: {test_loss:.4f}")
    print(f"Validation accuracy: {test_accuracy:.4f}")

    y_score = model.predict(validation_generator)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.asarray(validation_generator.classes)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    target_names = list(validation_generator.class_indices.keys())
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    plot_roc(y_true, y_score, target_names, output_dir / "roc_curve.png")

    metrics_path = output_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as fh:
        fh.write(f"validation_loss={test_loss:.6f}\n")
        fh.write(f"validation_accuracy={test_accuracy:.6f}\n")
        fh.write(f"micro_f1={micro_f1:.6f}\n")
        fh.write(f"macro_f1={macro_f1:.6f}\n")
        fh.write("\nConfusion Matrix:\n")
        fh.write(np.array2string(cm))
        fh.write("\n\n")
        fh.write(classification_report(y_true, y_pred, target_names=target_names))

    if args.save_model:
        model_path = output_dir / "dermareach_vgg16.keras"
        model.save(model_path)
        print(f"Saved model to {model_path}")

    print(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    # Avoid TensorFlow grabbing all GPU memory when other processes need it.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
