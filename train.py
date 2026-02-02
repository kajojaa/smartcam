from rfdetr import RFDETRMedium

# Create a model with a pretrained COCO backbone
# and specify number of classes = 1 (just basketball)
model = RFDETRMedium()

# Train on your dataset
model.train(
    dataset_dir="data/basketball",  # expects train/valid/test subdirs
    epochs=50,
    batch_size=2,
    lr=1e-4
)

# Save the fine‑tuned model weights
model.save("models/basketball-rfdetr.pth")
