from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights="models/basketball6b.pth")
model.optimize_for_inference()
#model.export()
model.export(simplify=True)