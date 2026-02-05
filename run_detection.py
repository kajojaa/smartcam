import os
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# -------------------------
# CONFIG
# -------------------------
IMAGE_DIR = "data/testset"
OUTPUT_DIR = "data/testset/detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = "models/basketball2k_01.pth"

# -------------------------
# LOAD MODEL
# -------------------------
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH)
model.optimize_for_inference()

# -------------------------
# SETUP SUPERVISION ANNOTATORS
# -------------------------
example_image = Image.open(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0]))
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
# RUN DETECTION
# -------------------------
for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(IMAGE_DIR, fname)
    image = Image.open(path)

    # Run RF-DETR
    detections = model.predict(image, threshold=0.5)

    # Build labels for visualization
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="data/basketball2k/test",
        annotations_path="data/basketball2k/test/_annotations.coco.json",
    )

    detections_labels = [
    f"{ds.classes[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id.tolist(), detections.confidence.tolist())
    ]

    # Annotate image
    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, detections_labels)

    # Save or show
    out_path = os.path.join(OUTPUT_DIR, fname)
    annotated_image.save(out_path)
    print(f"Saved: {out_path}")
