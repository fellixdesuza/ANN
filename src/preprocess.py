import os
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect label column in CIC-IDS style datasets (sometimes with leading space)."""
    candidates = [" Label", "Label", "label", "LABEL"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Label column not found. Tried: " + ", ".join(candidates))


def row_to_grayscale_image(row_vec: np.ndarray, img_size: int = 32) -> Image.Image:
    """
    Convert 1D scaled feature vector -> padded square -> grayscale PIL image resized to img_size.

    NOTE: row_vec must already be scaled to [0,1].
    """
    row_vec = np.asarray(row_vec, dtype=np.float32)

    n_features = row_vec.shape[0]
    side = int(math.ceil(math.sqrt(n_features)))
    square_len = side * side

    padded = np.zeros(square_len, dtype=np.float32)
    padded[:n_features] = row_vec[:square_len]

    img2d = padded.reshape(side, side)
    img_uint8 = (img2d * 255.0).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_uint8, mode="L")
    pil_img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    return pil_img


def make_dirs(base_out: str):
    for split in ["train", "test"]:
        for cls in ["benign", "ddos"]:
            os.makedirs(os.path.join(base_out, split, cls), exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="CSV -> grayscale PNG dataset generator")
    p.add_argument("--csv", required=True, help="Path to CIC-IDS CSV file")
    p.add_argument("--out", default="png_dataset", help="Output folder for PNG dataset")
    p.add_argument("--img-size", type=int, default=32, help="PNG size (e.g. 32, 64)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 = all)")
    p.add_argument("--scaler", default="models/scaler.joblib", help="Save scaler path")
    return p.parse_args()


def main():
    args = parse_args()

    print("[1] Loading CSV...")
    df = pd.read_csv(args.csv, nrows=None if args.max_rows == 0 else args.max_rows)

    label_col = detect_label_column(df)
    df[label_col] = df[label_col].astype(str)

    # Keep only BENIGN and DDoS (binary classification)
    df = df[df[label_col].isin(["BENIGN", "DDoS"])].copy()
    if df.empty:
        raise ValueError("No BENIGN/DDoS rows found. Check your CSV label values.")

    df["target"] = df[label_col].map({"BENIGN": 0, "DDoS": 1})
    df = df.drop(columns=[label_col])

    y = df["target"].values.astype(int)
    X = df.drop(columns=["target"])

    # Clean numeric values
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    print(f"[2] Samples: {len(X)} | Features: {X.shape[1]}")

    # Split first (avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(args.scaler), exist_ok=True)
    joblib.dump(scaler, args.scaler)
    print(f"[3] Scaler saved: {args.scaler}")

    # Output directories
    make_dirs(args.out)

    # Write PNGs
    print("[4] Writing TRAIN PNG images...")
    for i in range(len(X_train_scaled)):
        cls = "benign" if y_train[i] == 0 else "ddos"
        img = row_to_grayscale_image(X_train_scaled[i], img_size=args.img_size)
        img.save(os.path.join(args.out, "train", cls, f"{i:07d}.png"))

    print("[5] Writing TEST PNG images...")
    for i in range(len(X_test_scaled)):
        cls = "benign" if y_test[i] == 0 else "ddos"
        img = row_to_grayscale_image(X_test_scaled[i], img_size=args.img_size)
        img.save(os.path.join(args.out, "test", cls, f"{i:07d}.png"))

    print("\n✅ PNG dataset created successfully!")
    print(f"Output folder: {args.out}")
    print("Structure:")
    print(f"  {args.out}/train/benign/")
    print(f"  {args.out}/train/ddos/")
    print(f"  {args.out}/test/benign/")
    print(f"  {args.out}/test/ddos/")


if __name__ == "__main__":
    main()
