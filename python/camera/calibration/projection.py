import cv2
import numpy as np
import json
import numpy as np
from PIL import Image

import cv2
import numpy as np


def restore_image(image):
    cx, cy = 988, 509
    r = 560
    orig_w, orig_h = 1920, 1080
    crop_size = r * 2
    restored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = crop_size - max(0, x2 - orig_w)
    src_y2 = crop_size - max(0, y2 - orig_h)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(orig_w, x2)
    dst_y2 = min(orig_h, y2)

    restored[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return restored


json_file_path = "./outputs/calibration-1920-1080.json"

with open(json_file_path, "r") as file:  # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data["mtx"])
dst = np.array(json_data["dist"])

image_path = ".captured_images/test_images/image_1.png"
image = cv2.imread(image_path)
# image = restore_image(image)

h, w = image.shape[:2]
print((h, w))
image = cv2.undistort(image, mtx, dst, None, None)

cv2.imshow("Undistorted Image", cv2.resize(image, np.asarray([w,h])//2))
# cv2.imwrite(".captured_images/test_images/image_0-undistorted.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
