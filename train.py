from rfdetr import RFDETRBase

dataset = "data/basketball2k"

model = RFDETRBase()

model.train(
    dataset_dir = dataset,
    epochs = 16, 
    batch_size = 1, 
    grad_accum_steps = 16,
    lr = 1e-4,
    output_dir = "model2k",
    #resume="output/checkpoint.pth",
)