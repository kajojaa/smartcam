import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "data/testset/video.mov"
CHECKPOINT_PATH = "models/basketball2k_01.pth"
CONF_THRESHOLD = 0.5

# -------------------------
# LOAD MODEL
# -------------------------
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH)
model.optimize_for_inference()

# -------------------------
# LOAD DATASET CLASSES
# -------------------------
# This ensures labels match your training classes
ds = sv.DetectionDataset.from_coco(
    images_directory_path="data/basketball2k/test",
    annotations_path="data/basketball2k/test/_annotations.coco.json",
)

CLASS_NAMES = ds.classes  # e.g., ["basketball", "player"]

# -------------------------
# OPEN VIDEO
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

# Example frame for resolution-dependent annotation sizes
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")
example_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

text_scale = sv.calculate_optimal_text_scale(resolution_wh=example_image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=example_image.size)

bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_thickness=thickness,
    smart_position=True
)

# -------------------------
# PROCESS VIDEO FRAMES
# -------------------------
print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run detection
    detections = model.predict(image_pil, threshold=CONF_THRESHOLD)

    # Convert tensors to lists
    class_ids = detections.class_id.tolist()
    confidences = detections.confidence.tolist()

    # Build labels using your training dataset classes
    detections_labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)
    ]

    # Annotate frame
    annotated_image = image_pil.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, detections_labels)

    # Convert back to OpenCV BGR for display
    annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

    # Display
    cv2.namedWindow("RF-DETR Real-Time Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("RF-DETR Real-Time Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
