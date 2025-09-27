#!/usr/bin/env python3
"""
inference_save_video.py

Batched sliding-window inference for crack segmentation on video.
Saves a new video (same resolution & fps) with cracks overlaid in red.

Usage:
    python3 inference_save_video.py --input input.mp4 --output output.mp4 \
        --patch 448 --overlap 0.2 --batch 16 --threshold 0.5 --morph 3
"""

import argparse
import time
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--patch", type=int, default=448, help="Patch size (square) - matches training size")
    p.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction between 0 and <1")
    p.add_argument("--batch", type=int, default=16, help="Batch size for patch inference")
    p.add_argument("--threshold", type=float, default=0.2, help="Probability threshold for mask")
    p.add_argument("--morph", type=int, default=3, help="Morphological kernel size (0 to disable)")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if empty)")
    return p.parse_args()

# ---------------------------
# Patch-preprocess helper
# ---------------------------
def preprocess_patch(patch, normalize):
    # patch: HxWxC (numpy, RGB uint8)
    t = torch.from_numpy(patch).float().permute(2,0,1) / 255.0  # (C,H,W), float [0,1]
    t = normalize(t)
    return t

# ---------------------------
# sliding window batched inference
# ---------------------------
def sliding_window_inference_batched(frame, model, device, preprocess_fn,
                                     patch_size=448, overlap=0.5, batch_size=16,
                                     threshold=0.2, morph_kernel=3):
    """
    frame: HxWxBGR numpy (OpenCV frame)
    returns: binary mask HxW (uint8) where 1 indicates crack
    """
    # convert to RGB (model expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    stride = max(1, int(patch_size * (1 - overlap)))

    # compute coordinates so that edges are included
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] != max(0, h - patch_size):
        ys.append(max(0, h - patch_size))
    if xs[-1] != max(0, w - patch_size):
        xs.append(max(0, w - patch_size))

    full_mask = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patches = []
    coords = []

    # gather patches and run in batches
    for yy in ys:
        for xx in xs:
            patch = frame_rgb[yy:yy+patch_size, xx:xx+patch_size]
            # if frame smaller than patch, resize patch to patch_size (rare)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

            t = preprocess_fn(patch)  # tensor (C,H,W)
            patches.append(t)
            coords.append((yy, xx))

            if len(patches) >= batch_size:
                batch = torch.stack(patches).to(device)  # (B,C,H,W)
                with torch.no_grad():
                    preds = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]  # (B, H, W)

                for (cy, cx), pred in zip(coords, preds):
                    full_mask[cy:cy+patch_size, cx:cx+patch_size] += pred
                    count_map[cy:cy+patch_size, cx:cx+patch_size] += 1

                patches, coords = [], []

    # process leftovers
    if len(patches) > 0:
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
        for (cy, cx), pred in zip(coords, preds):
            full_mask[cy:cy+patch_size, cx:cx+patch_size] += pred
            count_map[cy:cy+patch_size, cx:cx+patch_size] += 1

    # normalize blended probabilities
    full_mask /= np.maximum(count_map, 1e-6)
    # threshold
    mask = (full_mask >= threshold).astype(np.uint8)

    # optional morphology to remove small noise / hole fill
    if morph_kernel and morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print("Using device:", device)

    # load model (match architecture used at training)
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,  # load your trained weights below
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    model.to(device)
    model.eval()

    # preprocessing normalize (ImageNet stats for resnet encoder)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps} FPS")

    # VideoWriter (same size & fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = sliding_window_inference_batched(
            frame,
            model=model,
            device=device,
            preprocess_fn=lambda p: preprocess_patch(p, normalize),
            patch_size=args.patch,
            overlap=args.overlap,
            batch_size=args.batch,
            threshold=args.threshold,
            morph_kernel=args.morph
        )

        # overlay (red)
        overlay = frame.copy()
        overlay[mask == 1] = (0, 0, 255)  # BGR red
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        out.write(blended)

        frame_idx += 1
        # print progress every 50 frames
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            avg_fps = frame_idx / elapsed
            print(f"Processed {frame_idx} frames — avg FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    total_time = time.time() - t0
    print(f"Done. Processed {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.2f} FPS).")
    print("Output saved to:", args.output)

if __name__ == "__main__":
    main()

