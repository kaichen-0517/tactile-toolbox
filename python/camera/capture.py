import cv2
import os

FOLDER = "./.captured_images/calibration-1920-1080-2"

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {int(actual_w)}x{int(actual_h)}")

if not cap.isOpened():
    print("Unable to open camera!")
    exit()

print("Instructions: Press 's' to save the image, press 'q' to exit the preview.")

img_counter = 0
os.makedirs(FOLDER, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    scale = min(800 / w, 600 / h, 1.0)

    show = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("img", show)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break
    elif key == ord("s"):
        img_name = f"{FOLDER}/captured_image_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"The image has been saved as: {img_name}")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
