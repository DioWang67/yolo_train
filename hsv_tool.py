# import cv2
# import numpy as np

# # 讀取圖像
# image = cv2.imread(r'D:\Git\robotlearning\yolo_train\output\images\Pcb0_aug_2.jpg')  # 請將這裡的 'path_to_your_image.jpg' 換成您的圖片路徑
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # 創建一個窗口
# cv2.namedWindow("HSV Range Tool")

# # 初始化滑動條的初始值
# def nothing(x):
#     pass

# # 創建 HSV 滑動條
# cv2.createTrackbar("Lower H", "HSV Range Tool", 0, 180, nothing)
# cv2.createTrackbar("Upper H", "HSV Range Tool", 180, 180, nothing)
# cv2.createTrackbar("Lower S", "HSV Range Tool", 0, 255, nothing)
# cv2.createTrackbar("Upper S", "HSV Range Tool", 255, 255, nothing)
# cv2.createTrackbar("Lower V", "HSV Range Tool", 0, 255, nothing)
# cv2.createTrackbar("Upper V", "HSV Range Tool", 255, 255, nothing)

# while True:
#     # 獲取滑動條的當前值
#     lower_h = cv2.getTrackbarPos("Lower H", "HSV Range Tool")
#     upper_h = cv2.getTrackbarPos("Upper H", "HSV Range Tool")
#     lower_s = cv2.getTrackbarPos("Lower S", "HSV Range Tool")
#     upper_s = cv2.getTrackbarPos("Upper S", "HSV Range Tool")
#     lower_v = cv2.getTrackbarPos("Lower V", "HSV Range Tool")
#     upper_v = cv2.getTrackbarPos("Upper V", "HSV Range Tool")

#     # 定義當前滑動條設置的 HSV 範圍
#     lower_hsv = np.array([lower_h, lower_s, lower_v])
#     upper_hsv = np.array([upper_h, upper_s, upper_v])

#     # 創建遮罩並應用
#     mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
#     result = cv2.bitwise_and(image, image, mask=mask)

#     # 顯示原圖和遮罩結果
#     # cv2.imshow("Original Image", image)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Filtered Image", result)

#     # 按下 'q' 鍵退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np

# 全局變量
ref_point = []
cropping = False

# 滑鼠回調函數，用於選取區域
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping
    
    # 按下滑鼠左鍵開始選取
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 當滑鼠左鍵保持按住的時候
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_image = image.copy()
            cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", temp_image)

    # 當釋放滑鼠左鍵時，完成選取
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # 畫出最終選取的矩形
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# 加載圖像
image = cv2.imread(r'D:\Git\robotlearning\yolo_inference\Result\20241127\FAIL\original\112005.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 按 'r' 重置圖像
    if key == ord("r"):
        image = clone.copy()

    # 按 'c' 進行 HSV 計算
    elif key == ord("c"):
        if len(ref_point) == 2:
            roi = hsv_image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            
            # 計算選取區域的 HSV 平均值及標準差
            mean_hsv = cv2.mean(roi)[:3]
            std_dev = cv2.meanStdDev(roi)[1].flatten()[:3]

            # 定義上下範圍
            lower_hsv = np.maximum([0, 0, 0], mean_hsv - std_dev * 1.5).astype(int)
            upper_hsv = np.minimum([180, 255, 255], mean_hsv + std_dev * 1.5).astype(int)

            # 調整格式為 [[H1, S1, V1], [H2, S2, V2]]
            lower_hsv_list = lower_hsv.tolist()
            upper_hsv_list = upper_hsv.tolist()
            hsv_range = [lower_hsv_list, upper_hsv_list]

            print("選取區域的 HSV 範圍:")
            print(f"範圍: {hsv_range}")

            # 使用這個範圍創建遮罩
            mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
            result = cv2.bitwise_and(image, image, mask=mask)

            # 顯示選取區域的結果
            cv2.imshow("Mask", mask)
            cv2.imshow("Filtered Image", result)
    
    # 按 'q' 鍵退出
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
