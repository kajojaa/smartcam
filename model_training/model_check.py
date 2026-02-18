import torch
import argparse

# -------------------------------
# 1) Allow argparse.Namespace (PyTorch 2.6+ safety)
# -------------------------------
torch.serialization.add_safe_globals([argparse.Namespace])

# -------------------------------
# 2) Load checkpoint SAFELY
# -------------------------------
ckpt_path = "models/basketball2k_01.pth"  # <-- change if needed
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

print("Checkpoint keys:", ckpt.keys())

# -------------------------------
# 3) Extract state_dict
# -------------------------------
if "model" not in ckpt:
    raise RuntimeError("Checkpoint does not contain 'model' key")

state_dict = ckpt["model"]

# -------------------------------
# 4) FAST CHECK: class count + background (NO model rebuild)
# -------------------------------
print("\n=== Fast inspection from state_dict ===")

class_keys = [k for k in state_dict.keys() if "class_embed" in k]

if not class_keys:
    print("Could not find class_embed in state_dict")
else:
    for k in class_keys:
        w = state_dict[k]
        print(f"{k}: shape = {tuple(w.shape)}")

        num_outputs = w.shape[0]
        print(f"→ classifier outputs: {num_outputs}")

        print("INTERPRETATION:")
        print(f"• If you trained on 4 classes:")
        print(f"  - {num_outputs} = 5 → background + 4 classes ✅")
        print(f"  - {num_outputs} = 4 → NO background ❌")

# -------------------------------
# 5) OPTIONAL: full forward pass (requires model code)
# -------------------------------
DO_FORWARD = True  # set False if you only want the fast check

if DO_FORWARD:
    print("\n=== Full forward pass inspection ===")

    # ⚠️ This import depends on the RF-DETR repo you used
    # Adjust if needed
    try:
        from rfdetr.models import build_model
    except ImportError as e:
        print("ERROR: Could not import build_model")
        print("If this fails, the fast check above is already enough.")
        raise e

    if "args" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'args' key")

    args = ckpt["args"]

    # Build model
    model = build_model(args)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input (match training resolution)
    dummy = torch.randn(1, 3, 416, 416)

    with torch.no_grad():
        outputs = model(dummy)

    print("Output type:", type(outputs))
    print("Output keys:", outputs.keys())

    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]

    print("pred_logits shape:", logits.shape)
    print("pred_boxes shape:", boxes.shape)

    # Softmax check
    probs = logits.softmax(-1)
    print("Unique predicted class indices:",
          torch.unique(probs.argmax(-1)))
