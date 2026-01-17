import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers
from src.model import build_cnn


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN using PNG dataset folders")
    p.add_argument("--data", default="png_dataset", help="Dataset folder (png_dataset)")
    p.add_argument("--img-size", type=int, default=32, help="PNG size (must match preprocessing)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--model-out", default="models/ddos_cnn_png.keras")
    p.add_argument("--history-out", default="models/training_history_png.json")
    return p.parse_args()


def main():
    args = parse_args()

    train_dir = os.path.join(args.data, "train")
    test_dir = os.path.join(args.data, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    print("[1] Loading PNG datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Normalize 0..255 -> 0..1
    norm = layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    print("[2] Building CNN model...")
    model = build_cnn(args.img_size)
    model.summary()

    print("[3] Training...")
    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"\n✅ Model saved: {args.model_out}")

    os.makedirs(os.path.dirname(args.history_out), exist_ok=True)
    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)
    print(f"✅ Training history saved: {args.history_out}")


if __name__ == "__main__":
    main()
