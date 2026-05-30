"""
AdaptiveThreshold 实时调参工具
拖动滑条 → 实时预览效果
按 S 保存当前参数，按 Q 退出
"""

import cv2
import numpy as np
import sys

# ─────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────
DEFAULT_BLOCK_SIZE = 11   # 必须奇数
DEFAULT_C          = 2
DEFAULT_METHOD     = 0    # 0=MEAN, 1=GAUSSIAN
DEFAULT_TYPE       = 0    # 0=BINARY, 1=BINARY_INV

# ─────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────
params = {
    "block_size": DEFAULT_BLOCK_SIZE,
    "C":          DEFAULT_C,
    "method":     DEFAULT_METHOD,
    "thresh_type": DEFAULT_TYPE,
    "blur_ksize": 0,   # 0=不模糊, 1=3x3, 2=5x5, 3=7x7
}

WIN_SRC    = "Source"
WIN_RESULT = "AdaptiveThreshold Result"
WIN_DIFF   = "Diff (MEAN vs GAUSSIAN)"

src_gray = None   # 全局灰度图


# ─────────────────────────────────────────
# 处理函数
# ─────────────────────────────────────────
def apply_threshold(gray, p):
    # 可选预模糊
    blur_map = {0: None, 1: (3,3), 2: (5,5), 3: (7,7)}
    ksize = blur_map[p["blur_ksize"]]
    if ksize:
        gray = cv2.GaussianBlur(gray, ksize, 0)

    method = cv2.ADAPTIVE_THRESH_MEAN_C if p["method"] == 0 \
             else cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    ttype  = cv2.THRESH_BINARY if p["thresh_type"] == 0 \
             else cv2.THRESH_BINARY_INV

    # block_size 必须为奇数且 >= 3
    bs = max(3, p["block_size"])
    if bs % 2 == 0:
        bs += 1

    result = cv2.adaptiveThreshold(
        gray, 255, method, ttype, bs, p["C"]
    )
    return result


def render():
    if src_gray is None:
        return

    result = apply_threshold(src_gray, params)

    # 同时计算两种方法做对比
    p_mean = dict(params, method=0)
    p_gauss = dict(params, method=1)
    bs = max(3, params["block_size"])
    if bs % 2 == 0: bs += 1
    mean_r  = cv2.adaptiveThreshold(src_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, bs, params["C"])
    gauss_r = cv2.adaptiveThreshold(src_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, bs, params["C"])
    diff = cv2.absdiff(mean_r, gauss_r)

    # 叠加参数文字
    info = [
        f"Method    : {'MEAN' if params['method']==0 else 'GAUSSIAN'}",
        f"BlockSize : {bs}",
        f"C         : {params['C']}",
        f"Type      : {'BINARY' if params['thresh_type']==0 else 'BINARY_INV'}",
        f"PreBlur   : {['None','3x3','5x5','7x7'][params['blur_ksize']]}",
    ]
    display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for i, line in enumerate(info):
        cv2.putText(display, line, (10, 25 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow(WIN_RESULT, display)
    # cv2.imshow(WIN_DIFF,   diff)


# ─────────────────────────────────────────
# 滑条回调
# ─────────────────────────────────────────
def on_block_size(val):
    # 滑条范围 1~100，映射到奇数 3~201
    bs = val * 2 + 3
    params["block_size"] = bs
    render()

def on_c(val):
    params["C"] = val - 50   # 滑条 0~100 → C: -50~50
    render()

def on_method(val):
    params["method"] = val
    render()

def on_type(val):
    params["thresh_type"] = val
    render()

def on_blur(val):
    params["blur_ksize"] = val
    render()


# ─────────────────────────────────────────
# 保存参数
# ─────────────────────────────────────────
def save_params():
    bs = max(3, params["block_size"])
    if bs % 2 == 0: bs += 1
    method_str = "cv2.ADAPTIVE_THRESH_MEAN_C" if params["method"] == 0 \
                 else "cv2.ADAPTIVE_THRESH_GAUSSIAN_C"
    type_str   = "cv2.THRESH_BINARY" if params["thresh_type"] == 0 \
                 else "cv2.THRESH_BINARY_INV"
    blur_map   = {0:"None", 1:"(3,3)", 2:"(5,5)", 3:"(7,7)"}

    code = f"""
# ── AdaptiveThreshold 调参结果 ──────────────────
import cv2

# 预模糊（可选）
blur_ksize = {blur_map[params['blur_ksize']]}
if blur_ksize:
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)

result = cv2.adaptiveThreshold(
    gray,
    maxValue   = 255,
    adaptiveMethod = {method_str},
    thresholdType  = {type_str},
    blockSize  = {bs},
    C          = {params['C']}
)
"""
    with open("best_params.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("\n✅ 参数已保存到 best_params.py")
    print(code)


# ─────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────
def main():
    global src_gray
    cap = cv2.VideoCapture(0)

    path = "temp/scripts/threshold/image_1.png"
    # path = None
    
    if len(sys.argv) > 1 or path is not None:
        # img = cv2.imread(sys.argv[1])
        img = cv2.imread(path)
        
        if img is None:
            print(f"无法读取图片")
            sys.exit(1)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 2]
        use_camera = False
    else:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        use_camera = True
        ret, frame = cap.read()
        if ret:
            # src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 2]

    # 创建窗口
    # cv2.namedWindow(WIN_SRC,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(WIN_DIFF,   cv2.WINDOW_NORMAL)

    # ── 创建滑条 ──────────────────────────────────
    # BlockSize: 滑条值 v → blockSize = v*2+3（保证奇数，范围 3~203）
    cv2.createTrackbar("BlockSize(x2+3)", WIN_RESULT,
                       (DEFAULT_BLOCK_SIZE - 3) // 2, 100, on_block_size)

    # C: 滑条 0~100，实际 C = val-50，范围 -50~50
    cv2.createTrackbar("C (+50offset)",   WIN_RESULT,
                       DEFAULT_C + 50, 100, on_c)

    # Method: 0=MEAN, 1=GAUSSIAN
    cv2.createTrackbar("Method(0M 1G)",   WIN_RESULT,
                       DEFAULT_METHOD, 1, on_method)

    # Type: 0=BINARY, 1=BINARY_INV
    cv2.createTrackbar("Type(0B 1Inv)",   WIN_RESULT,
                       DEFAULT_TYPE, 1, on_type)

    # PreBlur: 0=无 1=3x3 2=5x5 3=7x7
    cv2.createTrackbar("PreBlur(0-3)",    WIN_RESULT,
                       0, 3, on_blur)

    print("操作说明:")
    print("  拖动滑条 → 实时预览")
    print("  S        → 保存当前参数到 best_params.py")
    print("  Q / ESC  → 退出")

    render()

    while True:
        if use_camera:
            ret, frame = cap.read()
            if ret:
                src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 2]
                cv2.imshow(WIN_SRC, src_gray)
                render()
        # else:
        #     if src_gray is not None:
        #         cv2.imshow(WIN_SRC, src_gray)

        key = cv2.waitKey(16) & 0xFF   # ~60Hz
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            save_params()

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()