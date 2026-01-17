"""Utilities for converting a numeric feature row to a grayscale image tensor."""

import math
import numpy as np
from PIL import Image


def row_to_grayscale_tensor(row_vec: np.ndarray, img_size: int = 32) -> np.ndarray:
    """
    Convert 1D scaled feature vector -> padded square -> resized grayscale image tensor.

    Parameters
    ----------
    row_vec : np.ndarray
        1D vector of features (must be scaled to [0,1]).
    img_size : int
        Output image size (img_size x img_size).

    Returns
    -------
    np.ndarray
        Tensor with shape (1, img_size, img_size, 1) float32 normalized to [0,1].
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

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr.reshape(1, img_size, img_size, 1)
