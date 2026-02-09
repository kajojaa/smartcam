import cv2
import os

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "data/video/video.mov"
OUTPUT_DIR = "saved_frames"
CROP_W = 416	
CROP_H = 416
JPEG_QUALITY = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# STATE
# -------------------------
crop_x, crop_y = 100, 100
dragging = False
saved_index = 190
frame_index = 0
frame = None
redraw = True

# -------------------------
# MOUSE CALLBACK
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global crop_x, crop_y, dragging, redraw

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        crop_x = x - CROP_W // 2
        crop_y = y - CROP_H // 2
        redraw = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        crop_x = x - CROP_W // 2
        crop_y = y - CROP_H // 2
        redraw = True

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        redraw = True

# -------------------------
# VIDEO
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    redraw = True

    # -------- Per-frame interactive loop --------
    while True:
        if redraw:
            display = frame.copy()

            # Clamp crop inside frame
            cx = max(0, min(crop_x, w - CROP_W))
            cy = max(0, min(crop_y, h - CROP_H))

            cv2.rectangle(
                display,
                (cx, cy),
                (cx + CROP_W, cy + CROP_H),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Frame: {frame_index} | Saved: {saved_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("frame", display)
            redraw = False

        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        elif key in (ord('n'), ord(' ')):
            frame_index += 1
            break

        elif key == ord('s'):
            crop = frame[
                cy:cy + CROP_H,
                cx:cx + CROP_W
            ]

            out_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{saved_index:05d}.jpg"
            )

            cv2.imwrite(
                out_path,
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            saved_index += 1
            frame_index += 1
            break   # ← THIS is the key line: advance frame
    # --------------------------------------------

cap.release()
cv2.destroyAllWindows()
