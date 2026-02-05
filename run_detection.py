import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

import supervision as sv
from rfdetr import RFDETRBase

# ---------------- CONFIG ----------------
MODEL_PATH = "models/basketball2k_01.pth"
CONF_THRESHOLD = 0.5

# -------- File Picker --------
root = tk.Tk()
root.withdraw()

FILE_PATH = filedialog.askopenfilename(
    title="Select image or video",
    filetypes=[
        ("Images & Videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.mov *.avi *.mkv"),
        ("All files", "*.*"),
    ],
)

if not FILE_PATH:
    raise RuntimeError("No file selected")

# -------- Load Model --------
model = RFDETRBase(pretrain_weights=MODEL_PATH)
model.optimize_for_inference()

# -------- Load Class Names --------
CLASS_NAMES = ["basketballs", "basketball", "rim", "sports ball"]


# -------- Helpers --------
def run_detection(image_pil, bbox_annotator, label_annotator):
    detections = model.predict(image_pil, threshold=CONF_THRESHOLD)

    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(
            detections.class_id.tolist(),
            detections.confidence.tolist()
        )
    ]

    annotated = image_pil.copy()
    annotated = bbox_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    return cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

# -------- IMAGE MODE --------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

ext = os.path.splitext(FILE_PATH)[1].lower()

if ext in IMAGE_EXTS:
    image_pil = Image.open(FILE_PATH).convert("RGB")

    text_scale = sv.calculate_optimal_text_scale(image_pil.size)
    thickness = sv.calculate_optimal_line_thickness(image_pil.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True,
    )

    result = run_detection(image_pil, bbox_annotator, label_annotator)

    cv2.imshow("RF-DETR Detection (Image)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------- VIDEO MODE --------
elif ext in VIDEO_EXTS:
    cap = cv2.VideoCapture(FILE_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    example_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    text_scale = sv.calculate_optimal_text_scale(example_pil.size)
    thickness = sv.calculate_optimal_line_thickness(example_pil.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True,
    )

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_frame = run_detection(image_pil, bbox_annotator, label_annotator)

        cv2.namedWindow("RF-DETR Detection (Video)", cv2.WINDOW_NORMAL)
        cv2.imshow("RF-DETR Detection (Video)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    raise RuntimeError("Unsupported file type")
