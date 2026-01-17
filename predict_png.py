import argparse
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf
from src.inference import row_to_grayscale_tensor


def detect_label_column(df: pd.DataFrame):
    for c in [" Label", "Label", "label", "LABEL"]:
        if c in df.columns:
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Predict DDoS/BENIGN using PNG CNN model")
    p.add_argument("--model", required=True, help="Path to trained model .keras")
    p.add_argument("--png", default="", help="Path to grayscale PNG file")
    p.add_argument("--csv", default="", help="Path to CSV file")
    p.add_argument("--row", type=int, default=0, help="Row index for CSV prediction")
    p.add_argument("--scaler", default="models/scaler.joblib", help="Scaler path")
    p.add_argument("--img-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    print(f"[1] Loaded model: {args.model}")

    # Case A: Predict from PNG file
    if args.png:
        img = Image.open(args.png).convert("L").resize((args.img_size, args.img_size))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.reshape(1, args.img_size, args.img_size, 1)

        prob = float(model.predict(x, verbose=0)[0][0])
        pred = "DDoS" if prob >= 0.5 else "BENIGN"
        print(f"\n✅ PNG Prediction: {pred} | prob(DDoS)={prob:.6f}")
        return

    # Case B: Predict from CSV row
    if args.csv:
        df = pd.read_csv(args.csv)
        label_col = detect_label_column(df)
        if label_col and label_col in df.columns:
            df = df.drop(columns=[label_col])

        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        scaler = joblib.load(args.scaler)
        X_scaled = scaler.transform(df.values)

        if args.row < 0 or args.row >= len(X_scaled):
            raise IndexError(f"Row index out of range: 0..{len(X_scaled)-1}")

        x = row_to_grayscale_tensor(X_scaled[args.row], img_size=args.img_size)

        prob = float(model.predict(x, verbose=0)[0][0])
        pred = "DDoS" if prob >= 0.5 else "BENIGN"
        print(f"\n✅ CSV Row Prediction: row={args.row} -> {pred} | prob(DDoS)={prob:.6f}")
        return

    print("❌ You must provide either --png OR --csv")


if __name__ == "__main__":
    main()
