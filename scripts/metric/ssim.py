#!/usr/bin/env python3
"""Compute SSIM between two videos.

Usage:
    python scripts/metric/ssim.py dense.mp4 sparse.mp4 [--samples 16]
"""
import argparse
import numpy as np


def read_frames(path: str) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="Reference video (e.g. dense)")
    parser.add_argument("candidate", help="Candidate video (e.g. sparse)")
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    from skimage.metrics import structural_similarity
    a = read_frames(args.reference)
    b = read_frames(args.candidate)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape[1:] != b.shape[1:]:
        import cv2
        h, w = a.shape[1], a.shape[2]
        b = np.stack([cv2.resize(f, (w, h)) for f in b])

    s = args.samples
    indices = sorted({round(i * (n - 1) / (s - 1)) for i in range(min(s, n))})
    scores = [structural_similarity(a[i], b[i], channel_axis=2, data_range=255) for i in indices]
    print(f"SSIM: {np.mean(scores):.4f}  (frames sampled={len(indices)}/{n})")


if __name__ == "__main__":
    main()
