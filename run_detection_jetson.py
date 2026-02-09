import cv2
import numpy as np
import tensorrt as trt
import time

# ---------------- CONFIG ----------------
ENGINE_PATH = "basketball.trt"
VIDEO_PATH = "input1.mov"
CONF_THRESHOLD = 0.3

CLASS_NAMES = [
    "basketball",
    "rim",
    "sports ball",
]

# ---------------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# Get input / output info
input_binding = None
output_binding = None

for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        input_binding = i
    else:
        output_binding = i

input_shape = context.get_binding_shape(input_binding)
output_shape = context.get_binding_shape(output_binding)

_, _, H, W = input_shape

# Allocate buffers (Jetson unified memory)
input_buffer = np.empty(np.prod(input_shape), dtype=np.float32)
output_buffer = np.empty(np.prod(output_shape), dtype=np.float32)

bindings = [
    int(input_buffer.ctypes.data),
    int(output_buffer.ctypes.data),
]

# ---------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

fps = 0.0
alpha = 0.1

print("Running inference... Press ESC to exit")

while True:
    start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # Resize to TRT input size
    frame_resized = cv2.resize(frame, (W, H))

    # BGR -> RGB
    img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # HWC -> CHW, normalize to [0,1]
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    input_buffer[:] = img.ravel()

    # Run inference
    context.execute_v2(bindings)

    detections = output_buffer.reshape(-1, 6)

    # Draw detections
    for det in detections:
        x1, y1, x2, y2, score, class_id = det

        if score < CONF_THRESHOLD:
            continue

        x1 = int(x1 * frame.shape[1])
        x2 = int(x2 * frame.shape[1])
        y1 = int(y1 * frame.shape[0])
        y2 = int(y2 * frame.shape[0])

        cls = int(class_id)
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

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

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("RT-DETR TensorRT", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
