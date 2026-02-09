import os
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import supervision as sv

# ---------------- CONFIG ----------------
ENGINE_PATH = "basketball.trt"     # TensorRT engine path
VIDEO_FILE = "input1.MOV"       # video file in the same directory as script
CONF_THRESHOLD = 0.3
CLASS_NAMES = ["basketballs", "basketball", "rim", "sports ball"]

# ---------------- CHECK VIDEO FILE ----------------
if not os.path.exists(VIDEO_FILE):
    raise RuntimeError(f"Video file not found: {VIDEO_FILE}")

# ---------------- LOAD TENSORRT ENGINE ----------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# ---------------- ALLOCATE BUFFERS ----------------
inputs, outputs, bindings = [], [], []
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({"host": host_mem, "device": device_mem})
    else:
        outputs.append({"host": host_mem, "device": device_mem})

stream = cuda.Stream()

# ---------------- INFERENCE ----------------
def infer(frame):
    # Convert BGR → RGB, CHW, normalize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]  # (1,3,H,W)
    
    np.copyto(inputs[0]["host"], img_input.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    # parse output: N x 6 -> x1,y1,x2,y2,score,class
    out = outputs[0]["host"].reshape(-1, 6)
    mask = out[:, 4] >= CONF_THRESHOLD
    filtered = out[mask]

    if len(filtered) == 0:
        return None

    xyxy = filtered[:, :4]
    confidences = filtered[:, 4]
    class_ids = filtered[:, 5].astype(int)

    return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

# ---------------- ANNOTATION ----------------
def annotate(frame, detections):
    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id.tolist(), detections.confidence.tolist())
    ]
    text_scale = sv.calculate_optimal_text_scale(frame.shape[:2][::-1])
    thickness = sv.calculate_optimal_line_thickness(frame.shape[:2][::-1])
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    annotated = bbox_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    return annotated

# ---------------- RUN ----------------
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video {VIDEO_FILE}")

fps = 0.0
alpha = 0.1
print("Press ESC to exit")

while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    detections = infer(frame)
    annotated_frame = annotate(frame, detections) if detections is not None else frame

    # FPS calculation
    end_time = time.perf_counter()
    current_fps = 1.0 / (end_time - start_time)
    fps = (1 - alpha) * fps + alpha * current_fps
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("RFDETR (TensorRT) Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
