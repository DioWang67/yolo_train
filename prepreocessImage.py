import cv2
import numpy as np
import os

def preprocess_image(image_path, output_dir):
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return

    # 原圖保存
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), img)

    # 1. 轉灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "gray.jpg"), gray)

    # 2. 高斯模糊
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "blurred.jpg"), blurred)

    # 3. CLAHE 增強對比
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(output_dir, "clahe.jpg"), clahe_img)

    # 4. 邊緣檢測 (Canny)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(os.path.join(output_dir, "canny_edges.jpg"), edges)

    # 5. 二值化處理 (簡單閾值)
    _, binary_simple = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "binary_simple.jpg"), binary_simple)

    # 6. 自適應二值化
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(os.path.join(output_dir, "binary_adaptive.jpg"), binary_adaptive)

    # 7. 形態學操作 (開運算去除噪點)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, "morph_open.jpg"), morph_open)

    # 8. HSV 色域遮罩 (以紅色為例)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    result_red = cv2.bitwise_and(img, img, mask=mask_red)
    cv2.imwrite(os.path.join(output_dir, "mask_red.jpg"), result_red)

    print(f"圖片預處理完成，結果已儲存至資料夾: {output_dir}")

if __name__ == "__main__":
    # 指定圖片路徑與輸出資料夾
    image_path = r"D:\Git\robotlearning\yolo_inference_test\Result\20241211\FAIL\original\093331.jpg"  # 替換為你的圖片路徑
    output_dir = r"S:\DioWang\robotlearning\img"

    # 執行預處理
    preprocess_image(image_path, output_dir)
