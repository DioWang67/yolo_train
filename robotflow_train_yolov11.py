

# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="SvSDh2zS1kGJJX3ATwBK"
# )
# image_file = r"D:\Git\robotlearning\yoloface\target\result\data1\Pcb72.jpg"
# result = CLIENT.infer(image_file, model_id="cable-0h3zc/1")
# print(result)

import cv2
import requests
import numpy as np

# Roboflow API 客戶端設置
API_URL = "https://detect.roboflow.com"
API_KEY = "SvSDh2zS1kGJJX3ATwBK"
MODEL_ID = "cable-0h3zc/1"
IMAGE_SIZE = 640  # 推理時所需的圖像尺寸

# 設置攝像頭
def setup_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_SATURATION, 32)
    cap.set(cv2.CAP_PROP_HUE, 0)
    cap.set(cv2.CAP_PROP_SHARPNESS, 8)
    cap.set(cv2.CAP_PROP_GAMMA, 64)

# 從攝像頭捕捉圖像
def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        print("無法捕捉影像幀")
        return None
    return frame

# 發送圖像到 Roboflow API 並獲取推理結果
def infer_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)  # 編碼為JPG格式
    response = requests.post(
        f"{API_URL}/{MODEL_ID}",
        params={"api_key": API_KEY, "size": IMAGE_SIZE},
        files={"file": img_encoded.tobytes()},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return response.json()

# 繪製檢測框
def draw_detections(frame, detections):
    for detection in detections:
        x0, y0, x1, y1 = detection['x'], detection['y'], detection['x'] + detection['width'], detection['y'] + detection['height']
        conf = detection['confidence']
        class_name = detection['class']
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {conf:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# 主函數，用於從攝像頭捕獲影像並進行推理
def run_camera_inference():
    cap = cv2.VideoCapture(1)  # 使用默認攝像頭
    setup_camera(cap)
    
    if not cap.isOpened():
        print("無法打開攝像頭")
        return
    
    while True:
        frame = capture_image(cap)
        if frame is None:
            break

        # 將圖像發送到 Roboflow 進行推理
        result = infer_image(frame)
        print(result)  # 打印推理結果
        
        # 繪製檢測結果
        detections = result.get('predictions', [])
        draw_detections(frame, detections)
        
        # 顯示結果
        cv2.imshow('Roboflow 檢測', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference()
