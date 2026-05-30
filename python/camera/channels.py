import cv2
import numpy as np


def process(img, wait_key=0):
    # 1. 生成灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # COLOR_BGR2YUV

    # 2. 分离通道
    b, g, r = cv2.split(img)

    # 3. 拼接图像
    # hstack 是水平拼接（左右），vstack 是垂直拼接（上下）
    # 注意：所有拼接的图像必须维度一致。
    top_row = np.hstack((gray_img, b))   # 横向合并上方两张图
    bottom_row = np.hstack((g, r))       # 横向合并下方两张图
    combined = np.vstack((top_row, bottom_row)) # 纵向合并上下两行

    # 4. 调整窗口大小
    # 如果你的摄像头分辨率很高（如1080P），拼接后窗口会太大，可以缩小显示
    result_resized = cv2.resize(combined, (0, 0), fx=0.2, fy=0.2)

    # 5. 显示一个窗口
    cv2.imshow('All Channels (TopL:Gray, TopR:B, BotL:G, BotR:R)', result_resized)
    cv2.waitKey(wait_key)

    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # _, _, img = cv2.split(img)
    # ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("Original Gray", img)
    # cv2.imshow("Otsu Thresholding", thresh_img)
    # cv2.waitKey(wait_key)


img = cv2.imread("./temp/scripts/camera/image_0.png")
process(img, 0)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_EXPOSURE, -7)

# while 1:
#     ret, img = cap.read()
#     if not ret:
#         print("无法读取摄像头数据")
#     else:
#         process(img, 1)

# cap.release()

cv2.destroyAllWindows()
