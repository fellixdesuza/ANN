# CSV → Grayscale PNG Images → CNN (BENIGN vs DDoS)

## 1. Project Overview

This project converts **network traffic CSV data** into **grayscale PNG images** and then uses those images for **training** and **prediction** with a **Convolutional Neural Network (CNN)**.

The main idea is:

✅ **CSV rows (numeric features) → Grayscale images → CNN classifier**

This is deep learning systems approach, where CIC-IDS style network intrusion datasets are converted into images.

---

## 2. What This Project Does

### ✅ Input
A CSV dataset containing rows of network traffic features.
File used in this project: https://github.com/racsa-lab/Edge-Detect/blob/master/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Example dataset file:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### ✅ Output
1) A PNG image dataset folder structure:
```
png_dataset/
  train/
    benign/
    ddos/
  test/
    benign/
    ddos/
```

2) A trained CNN model:
- `models/ddos_cnn_png.keras`

3) Evaluation results:
- Confusion Matrix image: `models/confusion_matrix.png`
- Console output: Accuracy + Classification Report

4) Predictions:
- From a grayscale PNG file
- From a row in a CSV file (converted into grayscale tensor)

---

## 3. Dataset Information (Binary Classification)

Many CIC-IDS CSV datasets contain a label column named:

✅ `" Label"` (note the **leading space**)

This project performs **binary classification** using only:

| Label Value in CSV | Class Name | Target |
|-------------------|------------|--------|
| BENIGN            | benign     | 0 |
| DDoS              | ddos       | 1 |

If other attack types exist (PortScan, Bot, etc.), this project ignores them for this assignment.

---

## 4. How It Works (Technical Explanation)

### 4.1 CSV Row → Grayscale Image Conversion

Each CSV row is treated as **one sample**.

The CSV row contains a numeric feature vector like:

```
[f1, f2, f3, ..., f78]
```

The project converts this 1D vector into a 2D grayscale image using these steps:

#### Step 1 — Cleaning (NaN / Inf handling)
Some values may be invalid:
- NaN
- +inf / -inf

These are replaced with `0` to avoid errors.

#### Step 2 — Normalization (Min-Max Scaling)
Neural networks train better when data is scaled.

We use:

✅ `MinMaxScaler()`

This converts each feature value into the range:

```
[0, 1]
```

#### Step 3 — Padding into a square
CNN needs an image-like input (2D).  
But feature vectors are 1D.

For 78 features:
- √78 ≈ 8.8  
- Next square = 9×9 = 81

So the vector is padded with zeros:

```
78 → 81
```

#### Step 4 — Reshape into 2D
Now reshape:

```
(81,) → (9×9)
```

#### Step 5 — Convert to grayscale pixel values
Grayscale pixels are stored in:

```
0..255
```

So the normalized values are multiplied:

```
pixel = value * 255
```

#### Step 6 — Resize final image
A 9×9 image is too small for CNN to learn patterns well.

So the project resizes the image to:

✅ 32×32 (default)

Final image shape:

✅ `(32, 32, 1)` grayscale

---

### 4.2 Creating PNG Dataset for Training and Testing

After conversion, images are saved as real `.png` files inside:

```
png_dataset/train/benign/
png_dataset/train/ddos/
png_dataset/test/benign/
png_dataset/test/ddos/
```

Each CSV row becomes an image file like:

```
0000000.png
0000001.png
...
```

---

### 4.3 CNN Training Using PNG Images

Training uses TensorFlow’s function:

✅ `image_dataset_from_directory()`

This automatically loads PNG files and assigns labels based on folder name:

- benign → 0
- ddos → 1

The CNN architecture includes:
- Convolution Layers (Conv2D)
- MaxPooling
- Multiple convolution blocks
- Dense layers
- Dropout for regularization
- Sigmoid output for probability

Loss function:
✅ `binary_crossentropy`

Optimizer:
✅ `Adam`

Output:
✅ `probability of DDoS`

---

### 4.4 Prediction

This project supports **two prediction modes**:

#### ✅ Mode A: Predict from PNG file
Input is an existing grayscale PNG image.

Example:
```
png_dataset/test/ddos/0000001.png
```

CNN outputs probability:
- `>= 0.5` → DDoS
- `< 0.5` → BENIGN

#### ✅ Mode B: Predict from CSV row
Input is a CSV file + row index.

The row is:
- cleaned
- scaled using saved scaler
- converted into grayscale tensor
- predicted by the CNN

---

## 5. Project Structure

```
csv_image_cnn_ddos_png_project/
│
├── png_dataset/                 # created during preprocessing
│   ├── train/
│   │   ├── benign/
│   │   └── ddos/
│   └── test/
│       ├── benign/
│       └── ddos/
│
├── models/
│   ├── scaler.joblib            # MinMaxScaler saved for prediction
│   ├── ddos_cnn_png.keras       # trained CNN model
│   └── confusion_matrix.png     # evaluation result image
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # CSV → PNG dataset generator
│   ├── inference.py             # shared conversion utilities (optional)
│   └── model.py                 # CNN architecture reference (optional)
│
├── train_png.py                 # trains CNN using PNG dataset
├── evaluate_png.py              # evaluates model (report + confusion matrix)
├── predict_png.py               # predicts from PNG or CSV row
├── requirements.txt             # dependencies list
└── README.md                    # this documentation
```

---

## 6. Installation

### 6.1 Create Virtual Environment (Recommended)

#### Windows CMD
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / Mac
```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.2 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Project (Step-by-Step)

> ⚠️ WINDOWS NOTE  
> Do NOT use Linux-style `\` for multi-line commands.  
> Run commands in **one line** in Windows CMD.

---

### ✅ Step 1 — Convert CSV to PNG dataset

This command:
- loads CSV
- filters BENIGN + DDoS
- cleans NaN/Inf
- scales features
- creates grayscale PNG images
- splits train/test
- saves scaler

```cmd
python -m src.preprocess --csv "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --out png_dataset --img-size 32 --scaler models/scaler.joblib --max-rows 50000
```

#### Arguments explanation
- `--csv` → input dataset CSV file  
- `--out` → output folder for PNG dataset  
- `--img-size` → image resolution (32×32 recommended)  
- `--scaler` → scaler file saved for prediction  
- `--max-rows` → limit rows (optional, recommended for faster work)

✅ After this step you will have:
- PNG images in `png_dataset/`
- scaler file: `models/scaler.joblib`

---

### ✅ Step 2 — Train CNN model using PNG dataset

```cmd
python train_png.py --data png_dataset --img-size 32 --epochs 5 --batch-size 64 --model-out models/ddos_cnn_png.keras
```

✅ Output:
- trained model saved as:
```
models/ddos_cnn_png.keras
```

---

### ✅ Step 3 — Evaluate Model

This script prints:
- confusion matrix
- accuracy
- classification report

And saves:
- confusion matrix image: `models/confusion_matrix.png`

```cmd
python evaluate_png.py --data png_dataset --model models/ddos_cnn_png.keras --img-size 32
```

---

### ✅ Step 4A — Predict from a PNG image

```cmd
python predict_png.py --model models/ddos_cnn_png.keras --png png_dataset/test/ddos/0000001.png
```

Example output:
```
PNG Prediction: DDoS | prob(DDoS)=0.981234
```

---

### ✅ Step 4B — Predict from a CSV row

```cmd
python predict_png.py --model models/ddos_cnn_png.keras --csv "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" --row 100
```

Example output:
```
CSV Row Prediction: row=100 -> BENIGN | prob(DDoS)=0.032800
```

---

## 8. Methods Used in This Project

### ✅ Data Processing
- NaN / Inf handling
- Feature extraction
- Label encoding (BENIGN=0, DDoS=1)

### ✅ Normalization
- Min-Max Scaling using `MinMaxScaler()`

### ✅ Image Generation
- Vector padding
- Square reshape
- Grayscale pixel conversion
- Resizing to fixed image size

### ✅ Deep Learning Model
- CNN architecture (Conv2D + MaxPool)
- Dropout regularization
- Sigmoid output layer

### ✅ Evaluation
- Accuracy
- Confusion Matrix
- Precision / Recall / F1 Score (Classification Report)

---

## 9. Troubleshooting

### ✅ TensorFlow oneDNN messages
If you see messages like:

```
oneDNN custom operations are on...
```

✅ This is not an error.  
It is only an information log.

---

### ✅ Preprocessing is slow
This dataset is large (225k+ rows).

✅ Use:
```cmd
--max-rows 50000
```

---

### ✅ Windows command errors
If you use Linux command formatting:

```bash
python ... \
 --out ...
```

Windows will break it.

✅ Use commands in ONE LINE in CMD.

---

## 10. Future Improvements (Optional)

- Larger images (64×64)
- Class balancing if dataset is imbalanced
- Grad-CAM visualization for explainability
- Multi-class classification for more attack types

---

## 11. Summary

This project demonstrates a complete pipeline:

✅ CSV dataset → grayscale PNG conversion  
✅ CNN training using PNG dataset  
✅ Model evaluation (confusion matrix + report)  
✅ Predictions from PNG images or CSV rows


