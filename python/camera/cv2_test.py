import cv2
import time

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)

print(f"start...")
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
count = 0
t0 = time.time()

while count < 300:
    ret, frame = cap.read()
    if ret:
        count += 1

elapsed = time.time() - t0
print(f"{count/elapsed:.1f} fps")

cap.release()