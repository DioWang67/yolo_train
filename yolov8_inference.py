import cv2
import torch
from ultralytics import YOLO  # 导入 YOLOv8
import os
from utils.plots import Annotator, colors
# 預期的顏色順序
expected_color_order = ["Red", "Green", "Orange", "Yellow", "Black", "Black1"]

def preprocess_image(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).float().unsqueeze(0)
    im /= 255.0
    return im

def process_predictions(pred, frame, names):
    detections = []
    annotator = Annotator(frame, line_width=3, example=str(names))
    
    detected_colors = set()  # 使用集合來儲存已檢測的顏色

    for det in pred:
        if len(det):
            for xyxy, conf, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                color_name = names[int(cls)]
                
                # 如果該顏色已經出現過或不在預期順序中，則跳過
                if color_name in detected_colors or color_name not in expected_color_order:
                    continue

                # 加入檢測結果
                detections.append({
                    'label': color_name,
                    'confidence': conf.item(),
                    'box': [int(coord) for coord in xyxy]
                })

                # 設置框的顏色和標籤顏色
                color = colors(expected_color_order.index(color_name), True)
                
                # 顯示檢測框和標籤
                annotator.box_label(xyxy, f" {conf:.2f}", color=color)

                # 記錄該顏色，避免重複標籤
                detected_colors.add(color_name)

    # 根據預期順序排序檢測結果
    detections = sorted(detections, key=lambda d: expected_color_order.index(d['label']))

    return annotator.result(), detections

def draw_fixed_labels(frame, avg_scores):
    # 在左邊固定位置畫出固定順序的標籤
    label_offset_y = 30  # 每個標籤之間的垂直間距
    label_start_y = 30    # 第一個標籤的初始 y 位置
    label_x_pos = 10      # 標籤放在圖片左側的 x 位置

    for i, color_name in enumerate(expected_color_order):
        label_y_pos = label_start_y + i * label_offset_y
        
        # 如果顏色已被檢測，則顯示分數，否則顯示 "未檢測"
        score = avg_scores.get(color_name, 0)
        if score > 0:
            label_text = f"{color_name}: {score:.2f}"
        else:
            label_text = f"{color_name}: 未檢測"
        
        # 使用該顏色顯示標籤
        color = colors(i, True)
        cv2.putText(frame, label_text, (label_x_pos, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

def check_color_order(detections):
    detected_colors = [d['label'] for d in detections]
    return detected_colors == expected_color_order

def run_inference(weights='yolov8n.pt', source=0, device='cpu', conf_thres=0.25, iou_thres=0.45):
    model = YOLO(weights)  # 使用 YOLOv8 模型
    
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("無法打開攝像頭")
            return
        
        setup_camera(cap)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("無法捕捉影像幀")
                break
            
            results = model(frame)  # YOLOv8 直接进行推理
            result_frame, detections = process_predictions(results, frame, model.names)
            
            # 模擬多次檢測，這裡僅進行一次評估
            avg_scores = {det['label']: det['confidence'] for det in detections}
            
            # 繪製左側固定順序的標籤
            draw_fixed_labels(result_frame, avg_scores)
            
            # 檢查顏色順序
            is_correct_order = check_color_order(detections)
            order_text = "顏色順序正確" if is_correct_order else "顏色順序錯誤"

            # 顯示結果
            cv2.imshow('YOLOv8 檢測', result_frame)
            print(f"檢測結果: {detections}")
            print(f"顏色順序是否正確: {is_correct_order}")
            
            if is_correct_order:
                print("顏色順序正確，按下任意鍵繼續...")
                # 等待按下任意鍵後繼續
                cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(weights='yolov8n.pt', source=1, device='cpu', conf_thres=0.3, iou_thres=0.55)
