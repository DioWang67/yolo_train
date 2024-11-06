import cv2
import torch

# 加載 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 加載圖像
image_path = 'images/pic1.jpg'  # 替換為你的圖像路徑
img = cv2.imread(image_path)

# 進行檢測
results = model(img)

# 顯示檢測結果
results.show()
