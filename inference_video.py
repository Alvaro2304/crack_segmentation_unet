import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import time

# -------------------------------
# Load trained model
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet101",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load("model_final.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocessing (match training)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Sliding window with batching
# -------------------------------
def sliding_window_inference_batched(frame, patch_size=448, overlap=0.5, batch_size=16):
    h, w, _ = frame.shape
    stride = int(patch_size * (1 - overlap))

    full_mask = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patches, coords = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = frame[y:y+patch_size, x:x+patch_size]
            patches.append(preprocess(patch))
            coords.append((y, x))

            # Run when batch is full
            if len(patches) == batch_size:
                batch = torch.stack(patches).to(DEVICE)
                with torch.no_grad():
                    preds = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]

                for (yy, xx), pred in zip(coords, preds):
                    full_mask[yy:yy+patch_size, xx:xx+patch_size] += pred
                    count_map[yy:yy+patch_size, xx:xx+patch_size] += 1

                patches, coords = [], []

    # Process leftover patches
    if patches:
        batch = torch.stack(patches).to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]

        for (yy, xx), pred in zip(coords, preds):
            full_mask[yy:yy+patch_size, xx:xx+patch_size] += pred
            count_map[yy:yy+patch_size, xx:xx+patch_size] += 1

    full_mask /= np.maximum(count_map, 1e-6)
    return (full_mask > 0.2).astype(np.uint8)

# -------------------------------
# Video Inference
# -------------------------------
cap = cv2.VideoCapture("road.mp4")  # or 0 for webcam

fps_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    mask = sliding_window_inference_batched(frame, patch_size=448, overlap=0.2, batch_size=8)

    # Overlay cracks
    overlay = frame.copy()
    overlay[mask == 1] = (0, 0, 255)  # red cracks
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # FPS counter
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(blended, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show
    cv2.imshow("Crack Detection", blended)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

