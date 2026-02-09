import cv2
import numpy as np
import tensorrt as trt
import time

ENGINE_PATH = "basketball.trt"
VIDEO_PATH = "input1.mov"
CONF_THRESHOLD = 0.3

CLASS_NAMES = [
    "basketball",
    "rim",
    "sports ball",
]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---------------- Load engine ----------------
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# ---------------- Tensor info ----------------
input_name = None
output_name = None

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)

    if mode == trt.TensorIOMode.INPUT:
        input_name = name
    else:
        output_name = name

assert input_name and output_name

input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)

_, _, H, W = input_shape

# ---------------- Buffers (unified memory) ----------------
input_buffer = np.empty(np.prod(input_shape), dtype=np.float32)
output_buffer = np.empty(np.prod(output_shape), dtype=np.float32)

context.set_tensor_address(input_name, input_buffer.ctypes.data)
context.set_tensor_address(output_name, output_buffer.ctypes.data)

# ---------------- Video ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

fps = 0.0
alpha = 0.1

print("Running TensorRT inference — ESC to quit")

while True:
    start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # Resize to model resolution
    frame_resized = cv2.resize(frame, (W, H))

    # BGR → RGB
    img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # HWC → CHW, normalize
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_buffer[:] = img.ravel()

    # Inference
    context.execute_async_v3(0)

    detections = output_buffer.reshape(-1, 6)

    print(detections[:5])

    # Draw boxes
    for det in detections:
        cx, cy, w, h, score, cls_id = det
        if score < CONF_THRESHOLD:
            continue

        # Convert normalized cx,cy,w,h → pixel corners
        cx *= frame.shape[1]
        cy *= frame.shape[0]
        w  *= frame.shape[1]
        h  *= frame.shape[0]

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        cls_id = int(cls_id)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # FPS
    end = time.perf_counter()
    curr_fps = 1.0 / (end - start)
    fps = fps * (1 - alpha) + curr_fps * alpha

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("RT-DETR TensorRT", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
