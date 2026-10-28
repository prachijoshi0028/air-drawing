import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ---------- 1. SETTINGS ----------
# Colors are in BGR order (Blue, Green, Red) - that's how OpenCV works.
COLORS = [
    ("Pink",   (255, 0, 255)),
    ("Blue",   (255, 100, 0)),
    ("Green",  (0, 255, 0)),
    ("Yellow", (0, 255, 255)),
    ("Eraser", (0, 0, 0)),     # black = "nothing" on our canvas
]
HEADER_HEIGHT = 100   # height of the color bar at the top (pixels)
BRUSH_SIZE = 8        # thickness of the pen
ERASER_SIZE = 50      # thickness of the eraser
SMOOTHING = 5         # average the last N finger positions (less shaky lines)

# ---------- 2. SET UP MEDIAPIPE (the hand-tracking AI) ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------- 3. OPEN THE WEBCAM ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
current_color_index = 0
prev_point = None
recent_points = deque(maxlen=SMOOTHING)


def fingers_up(points):
    # A finger is "up" if its tip is higher on screen than its middle joint.
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    return [points[t][1] < points[j][1] for t, j in zip(tips, joints)]


def draw_header(frame, selected_index):
    h, w, _ = frame.shape
    box_w = w // len(COLORS)
    for i, (name, color) in enumerate(COLORS):
        x1, x2 = i * box_w, (i + 1) * box_w
        fill = color if name != "Eraser" else (60, 60, 60)
        cv2.rectangle(frame, (x1, 0), (x2, HEADER_HEIGHT), fill, -1)
        thickness = 6 if i == selected_index else 1
        cv2.rectangle(frame, (x1, 0), (x2, HEADER_HEIGHT), (255, 255, 255), thickness)
        cv2.putText(frame, name, (x1 + 20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ---------- 4. MAIN LOOP ----------
while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam. Is it connected / allowed?")
        break

    frame = cv2.flip(frame, 1)  # mirror the image
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        index_up, middle_up, ring_up, pinky_up = fingers_up(points)

        # smooth the index fingertip position (landmark 8)
        recent_points.append(points[8])
        x = int(np.mean([p[0] for p in recent_points]))
        y = int(np.mean([p[1] for p in recent_points]))

        # CLEAR: open palm (all four fingers up)
        if index_up and middle_up and ring_up and pinky_up:
            canvas[:] = 0
            prev_point = None

        # SELECT: index + middle fingers up
        elif index_up and middle_up:
            prev_point = None
            cv2.circle(frame, (x, y), 15, COLORS[current_color_index][1], -1)
            if y < HEADER_HEIGHT:
                box_w = w // len(COLORS)
                current_color_index = min(x // box_w, len(COLORS) - 1)

        # DRAW: only index finger up
        elif index_up and not middle_up:
            if y > HEADER_HEIGHT:
                name, color = COLORS[current_color_index]
                size = ERASER_SIZE if name == "Eraser" else BRUSH_SIZE
                cv2.circle(frame, (x, y), size // 2 + 4,
                           color if name != "Eraser" else (255, 255, 255), 2)
                if prev_point is None:
                    prev_point = (x, y)
                cv2.line(canvas, prev_point, (x, y), color, size)
                prev_point = (x, y)
            else:
                prev_point = None

        else:
            prev_point = None
    else:
        prev_point = None
        recent_points.clear()

    # put the drawing on top of the camera picture
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas[mask > 0]

    draw_header(frame, current_color_index)
    cv2.imshow("Air Drawing (q = quit, c = clear, s = save)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
    elif key == ord("s"):
        out = np.full_like(canvas, 255)
        m = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 10
        out[m] = canvas[m]
        cv2.imwrite("drawing.png", out)
        print("Saved drawing.png")

cap.release()
cv2.destroyAllWindows()