#!/usr/bin/env python3
"""Compute PSNR between two videos.

Usage:
    python scripts/metric/psnr.py dense.mp4 sparse.mp4
"""
import argparse
import math
import numpy as np


def read_frames(path: str) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="Reference video (e.g. dense)")
    parser.add_argument("candidate", help="Candidate video (e.g. sparse)")
    args = parser.parse_args()

    a = read_frames(args.reference)
    b = read_frames(args.candidate)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape[1:] != b.shape[1:]:
        import cv2
        h, w = a.shape[1], a.shape[2]
        b = np.stack([cv2.resize(f, (w, h)) for f in b])

    mse = np.mean((a - b) ** 2)
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    print(f"PSNR: {psnr:.4f} dB  (frames={n}, mse={mse:.6f})")


if __name__ == "__main__":
    main()
