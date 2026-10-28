from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

__all__ = ["load_view", "create_mask", "detect_top_boundary", "ultrasound_crop"]


def _read_bgr(path: str | Path) -> np.ndarray:
    with open(path, "rb") as handle:
        buffer = np.frombuffer(handle.read(), dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img


def create_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closing

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(closing)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def detect_top_boundary(image: np.ndarray, mask: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, _w = gray.shape

    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return 0

    safe_start_y = rows[len(rows) // 4]

    row_variances = []
    for y in range(h):
        row_mask = mask[y, :]
        if np.sum(row_mask) > 0:
            row_variances.append(np.var(gray[y, row_mask > 0]))
        else:
            row_variances.append(0)

    top_y = safe_start_y
    for y in range(safe_start_y, 0, -1):
        curr_var = row_variances[y]
        prev_var = row_variances[y - 1]
        if curr_var > 0:
            var_ratio = prev_var / curr_var
            if var_ratio > 2.5 or var_ratio < 0.4:
                top_y = y
                break
        if np.sum(mask[y - 1, :]) < np.sum(mask[y, :]) * 0.7:
            top_y = y
            break

    return top_y


def ultrasound_crop(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[0], img.shape[1]
    mask = create_mask(img)
    top_y = detect_top_boundary(img, mask)

    if abs(w - 1008) < abs(w - 1024):
        left_x, right_x = 40, 833
    else:
        left_x, right_x = 60, w - 60

    if abs(w - 1024) > abs(w - 1164):
        left_x, right_x = 60, w - 140

    bottom_y = h - 25
    cropped = img[top_y:bottom_y, left_x:right_x]
    return cropped if cropped.size > 0 else None


def load_view(path: str | Path, *, crop: str = "none") -> Image.Image:
    img = _read_bgr(path)

    if crop == "ultrasound":
        cropped = ultrasound_crop(img)
        if cropped is not None:
            img = cropped

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))
