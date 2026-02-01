import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------- CONFIG ----------
GRID_SIZE = 16
PINCH_THRESHOLD = 0.03    # thumb–index distance (normalized)
FIST_THRESHOLD = 0.02     # index–middle distance (normalized)
UPDATE_FRAMES = 3         # redraw 3D view every N frames

# ---------- MEDIAPIPE SETUP ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- VOXEL GRID ----------
voxels = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
colors = np.zeros(voxels.shape + (3,), dtype=float)  # RGB 0–1


def world_from_norm(x, y, z):
    """Map normalized hand coordinates (0–1) into discrete voxel indices."""
    ix = int((x - 0.1) / 0.8 * GRID_SIZE)
    iy = int((1 - y - 0.1) / 0.8 * GRID_SIZE)
    iz = int(z * GRID_SIZE * 0.5)
    ix = np.clip(ix, 0, GRID_SIZE - 1)
    iy = np.clip(iy, 0, GRID_SIZE - 1)
    iz = np.clip(iz, 0, GRID_SIZE - 1)
    return ix, iy, iz


def update_voxel(ix, iy, iz, create=True):
    """Create or erase a voxel at the given grid index."""
    if create:
        voxels[ix, iy, iz] = True
        colors[ix, iy, iz] = np.array([0.0, 1.0, 1.0])  # cyan
    else:
        voxels[ix, iy, iz] = False
        colors[ix, iy, iz] = 0


# ---------- GESTURE HELPERS ----------
def dist2d(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def classify_gesture(lm_list):
    """
    Very simple static gesture classifier:
    - pinch  : thumb–index close
    - fist   : index–middle close
    - open   : default
    """
    if not lm_list:
        return "none"

    lm_obj = lm_list[0]         # NormalizedLandmarkList
    pts = lm_obj.landmark       # list[NormalizedLandmark]

    thumb_tip = pts[4]
    index_tip = pts[8]
    middle_tip = pts[12]

    pinch_d = dist2d(thumb_tip, index_tip)
    spread = dist2d(index_tip, middle_tip)

    if pinch_d < PINCH_THRESHOLD:
        return "pinch"
    if spread < FIST_THRESHOLD:
        return "fist"
    return "open"


# ---------- 3D VIEW (MATPLOTLIB VOXELS) ----------
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')


def redraw_voxels():
    """Redraw the current voxel grid."""
    ax.clear()
    # transpose to match axes nicely
    ax.voxels(
        voxels.transpose((2, 1, 0)),
        facecolors=colors.transpose((2, 1, 0, 3)),
        edgecolor='k'
    )
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_title("Gesture‑Driven Voxel Canvas (pinch=draw, fist=erase)")
    plt.draw()
    plt.pause(0.001)


# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)
frame_count = 0

print("Controls:")
print("  Pinch thumb + index  = draw voxel")
print("  Fist (index + middle) = erase voxel")
print("  Press 'q' in the OpenCV window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    lm_list = result.multi_hand_landmarks

    gesture = classify_gesture(lm_list)

    if lm_list:
        lm_obj = lm_list[0]
        pts = lm_obj.landmark
        index_tip = pts[8]

        ix, iy, iz = world_from_norm(index_tip.x, index_tip.y, index_tip.z)

        if gesture == "pinch":
            update_voxel(ix, iy, iz, create=True)
        elif gesture == "fist":
            update_voxel(ix, iy, iz, create=False)

        # draw landmarks on the preview frame
        mp_drawing.draw_landmarks(
            frame, lm_obj, mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame, f"Gesture: {gesture}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    cv2.imshow("Webcam - Gesture Input", frame)

    frame_count += 1
    if frame_count % UPDATE_FRAMES == 0:
        redraw_voxels()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
