"""
draw_circles.py
---------------
Interactive tool to draw multiple circles on an image.

Controls
--------
  Left-click + drag  : draw a new circle (center = press point, radius = drag distance)
  Right-click        : delete the circle under the cursor
  'u'                : undo last circle
  'c'                : clear all circles
  's'                : save annotated image
  'q' / Esc          : quit
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATH = "./.captured_images/test_images/image_3.png"
WINDOW_NAME = "Draw Circles"
MAX_W, MAX_H = 1400, 900

CIRCLE_COLOR  = (0, 255, 0)   # drawn circles
PREVIEW_COLOR = (0, 200, 255) # circle being dragged
TEXT_COLOR    = (255, 255, 255)
LABEL_BG      = (0, 0, 0)
LINE_THICKNESS = 2
POINT_RADIUS   = 4

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
circles: list[tuple[int, int, int]] = []   # (cx, cy, r) in original-image coords
drawing = False
start_x, start_y = 0, 0
current_preview: tuple[int, int, int] | None = None

img_orig = None
SCALE = 1.0


def get_display_scale(img, max_w=MAX_W, max_h=MAX_H) -> float:
    h, w = img.shape[:2]
    return min(max_w / w, max_h / h, 1.0)


def to_orig(x, y) -> tuple[int, int]:
    return int(x / SCALE), int(y / SCALE)


def render(preview: tuple[int, int, int] | None = None) -> None:
    """Redraw the image with all committed circles and an optional preview."""
    canvas = img_orig.copy()

    for idx, (cx, cy, r) in enumerate(circles):
        _draw_circle_annotated(canvas, cx, cy, r, CIRCLE_COLOR, idx + 1)

    if preview:
        cx, cy, r = preview
        if r > 0:
            _draw_circle_annotated(canvas, cx, cy, r, PREVIEW_COLOR, len(circles) + 1)

    disp_w = int(canvas.shape[1] * SCALE)
    disp_h = int(canvas.shape[0] * SCALE)
    cv2.imshow(WINDOW_NAME, cv2.resize(canvas, (disp_w, disp_h)))


def _draw_circle_annotated(img, cx: int, cy: int, r: int, color, label: int) -> None:
    cv2.circle(img, (cx, cy), r, color, LINE_THICKNESS)
    cv2.circle(img, (cx, cy), POINT_RADIUS, color, -1)

    text = f"#{label}  c=({cx},{cy})  r={r}px"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

    tx = cx - tw // 2
    ty = cy - r - 8
    if ty - th - baseline < 0:
        ty = cy + r + th + 8

    # Background rectangle for readability
    cv2.rectangle(img, (tx - 2, ty - th - baseline), (tx + tw + 2, ty + baseline),
                  LABEL_BG, -1)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1,
                cv2.LINE_AA)


def circle_hit(cx: int, cy: int, r: int, px: int, py: int, tol: int = 8) -> bool:
    """True if point (px,py) is near the circumference of the circle."""
    dist = np.hypot(px - cx, py - cy)
    return abs(dist - r) < tol


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_preview, circles

    ox, oy = to_orig(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = ox, oy

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        r = int(np.hypot(ox - start_x, oy - start_y))
        current_preview = (start_x, start_y, r)
        render(current_preview)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        r = int(np.hypot(ox - start_x, oy - start_y))
        if r > 2:
            circles.append((start_x, start_y, r))
            print(f"[+] Circle #{len(circles)}: center=({start_x},{start_y}), radius={r}px")
        current_preview = None
        render()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete circle whose circumference is closest to the click
        for i in range(len(circles) - 1, -1, -1):
            cx, cy, r = circles[i]
            if circle_hit(cx, cy, r, ox, oy):
                removed = circles.pop(i)
                print(f"[-] Removed circle #{i+1}: center=({removed[0]},{removed[1]}), radius={removed[2]}px")
                render()
                break


def main():
    global img_orig, SCALE

    img_orig = cv2.imread(PATH)
    if img_orig is None:
        print(f"Failed to load image: {PATH}")
        return

    SCALE = get_display_scale(img_orig)
    disp_w = int(img_orig.shape[1] * SCALE)
    disp_h = int(img_orig.shape[0] * SCALE)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print(__doc__)
    render()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):   # q or Esc
            break

        elif key == ord('u'):       # undo
            if circles:
                removed = circles.pop()
                print(f"[u] Undo — removed circle: center=({removed[0]},{removed[1]}), radius={removed[2]}px")
                render()

        elif key == ord('c'):       # clear all
            circles.clear()
            print("[c] All circles cleared.")
            render()

        elif key == ord('s'):       # save
            save_path = os.path.splitext(PATH)[0] + "_annotated.png"
            canvas = img_orig.copy()
            for idx, (cx, cy, r) in enumerate(circles):
                _draw_circle_annotated(canvas, cx, cy, r, CIRCLE_COLOR, idx + 1)
            cv2.imwrite(save_path, canvas)
            print(f"[s] Saved to: {save_path}")

    cv2.destroyAllWindows()

    # Final summary
    if circles:
        print("\n=== Circle Summary ===")
        for i, (cx, cy, r) in enumerate(circles):
            print(f"  #{i+1}: center=({cx},{cy}), radius={r}px")
            
    print(circles)


if __name__ == "__main__":
    main()
