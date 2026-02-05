import supervision as sv

ds = sv.DetectionDataset.from_coco(
	images_directory_path=f"data/basketball2k/test",
	annotations_path=f"data/basketball2k/test/_annotations.coco.json",
)

path, image, annotations = ds[4]

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import numpy as np
import torch
from PIL import Image

image = Image.open(path)

CHECKPOINT_PATH = "models/basketball2k_01.pth"

model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH)

# Set to evaluation mode
#model.eval()
model.optimize_for_inference()

detections = model.predict(image, threshold=0.5)

text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
	text_color=sv.Color.BLACK,
	text_scale=text_scale,
	text_thickness=thickness,
	smart_position=True)

annotations_labels = [
	f"{ds.classes[class_id]}"
	for class_id
	in annotations.class_id
]

detections_labels = [
	f"{ds.classes[class_id]} {confidence:.2f}"
	for class_id, confidence
	in zip(detections.class_id, detections.confidence)
]

annotation_image = image.copy()
annotation_image = bbox_annotator.annotate(annotation_image, annotations)
annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

detections_image = image.copy()
detections_image = bbox_annotator.annotate(detections_image, detections)
detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1, 2), titles=["Annotation", "Detection"])