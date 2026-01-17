import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained CNN on PNG test dataset")
    p.add_argument("--data", default="png_dataset", help="Dataset folder")
    p.add_argument("--model", required=True, help="Path to trained model (.keras)")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cm-out", default="models/confusion_matrix_png.png")
    return p.parse_args()


def main():
    args = parse_args()

    test_dir = os.path.join(args.data, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    model = tf.keras.models.load_model(args.model)
    print(f"[1] Loaded model: {args.model}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Normalize
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

    # Collect y_true and predictions
    y_true = []
    y_prob = []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().ravel().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    print("\n[2] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["BENIGN", "DDoS"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("[3] Confusion Matrix:\n", cm)

    # Plot confusion matrix
    os.makedirs(os.path.dirname(args.cm_out), exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["BENIGN", "DDoS"])
    plt.yticks(tick_marks, ["BENIGN", "DDoS"])

    # Print values on cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(args.cm_out, dpi=200)
    plt.close()

    print(f"\n✅ Confusion matrix saved: {args.cm_out}")


if __name__ == "__main__":
    main()
