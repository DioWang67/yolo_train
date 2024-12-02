import cv2
import numpy as np

# 讀取圖片並轉換到 HSV
image = cv2.imread('Pcb0.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義顏色範圍
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([2, 255, 255])
lower_red2 = np.array([90, 100, 100])
upper_red2 = np.array([240, 255, 255])

lower_green = np.array([91, 73, 40])
upper_green = np.array([126, 255, 127])



lower_orange = np.array([5, 100, 100])
upper_orange = np.array([12, 255, 255])

lower_yellow = np.array([12, 100, 100])
upper_yellow = np.array([25, 255, 255])


lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# 建立遮罩
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# 提取顏色區域
result_red = cv2.bitwise_and(image, image, mask=mask_red)
result_green = cv2.bitwise_and(image, image, mask=mask_green)
result_black = cv2.bitwise_and(image, image, mask=mask_black)
result_orange = cv2.bitwise_and(image, image, mask=mask_orange)
result_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)

# 顯示結果
# cv2.imshow("Original Image", image)
cv2.imshow("Red Region", result_red)
cv2.imshow("Green Region", result_green)
# cv2.imshow("Black Region", result_black)
# cv2.imshow("Orange Region", result_orange)
# cv2.imshow("Yellow Region", result_yellow)

cv2.waitKey(0)
cv2.destroyAllWindows()
