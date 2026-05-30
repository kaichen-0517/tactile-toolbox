import cv2

# 1. 读取图像并转为灰度（Otsu 必须在单通道灰度图上运行）
img = cv2.imread(r'scripts\threshold\122.jpg', 0)

if img is None:
    print("错误：图片加载失败")
else:
    # 2. 调用 Otsu 二值化
    # 参数 0：在 Otsu 模式下，这个初始阈值会被忽略
    # 参数 255：最大值（超过阈值的像素设为 255）
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU：组合标志
    ret, thresh_img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    print(f"Otsu 自动计算出的最优阈值是: {ret}")

    # 3. 显示结果
    cv2.imshow('Original Gray', img)
    cv2.imshow('Otsu Thresholding', thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
