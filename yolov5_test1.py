import cv2
import torch
import os
import numpy as np
from utils.general import non_max_suppression, scale_boxes
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors

# 絕對路徑指定 default.yaml 的位置
config_path = 'D:/Git/robotlearning/yoloface/env/Lib/site-packages/ultralytics/cfg/default.yaml'

# 預期的顏色順序
expected_color_order = ["Red", "Green", "Orange", "Yellow", "Black", "Black1"]

# 設定判斷正確的分數閾值
confidence_threshold = 0.5

# 設定多次判斷的次數
evaluation_times = 20

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def load_model(weights, device, config_path):
    model = DetectMultiBackend(weights, device=device, data=config_path)
    stride, names = model.stride, model.names
    imgsz = (640, 640)
    return model, stride, names, imgsz

def preprocess_image(frame, device, imgsz):
    im, ratio, pad = letterbox(frame, imgsz, stride=32, auto=True)
    im = im.transpose((2, 0, 1))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float()
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)
    return im, ratio, pad

def process_predictions(pred, im, frame, names, conf_thres, iou_thres, ratio, pad):
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    detections = []
    annotator = Annotator(frame, line_width=3, example=str(names))
    
    detected_colors = set()  # 使用集合來儲存已檢測的顏色
    overlapping_threshold = 0.5  # 設置重疊的 IoU 閾值

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                color_name = names[int(cls)]
                
                # 如果該顏色已經出現過，則跳過
                if color_name in detected_colors:
                    continue

                # 檢查是否與已有的框重疊過多
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                overlapping = False
                for det_box in detections:
                    if iou(box, det_box['box']) > overlapping_threshold:
                        overlapping = True
                        break

                # 如果重疊過多，則跳過該框
                if overlapping:
                    continue

                # 加入檢測結果
                detections.append({
                    'label': color_name,
                    'confidence': conf.item(),
                    'box': box
                })

                # 設置框的顏色和標籤顏色
                color = colors(int(cls), True)
                
                # 顯示檢測框和標籤
                annotator.box_label(xyxy, color=color)

                # 記錄該顏色，避免重複標籤
                detected_colors.add(color_name)

    # 根據 x_center 排序，模擬從左到右的順序
    detections = sorted(detections, key=lambda d: d['box'][0])

    return annotator.result(), detections

def iou(box1, box2):
    """
    計算兩個框之間的 IoU 值 (Intersection over Union)
    box1 和 box2 格式為 [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 計算重疊部分的面積
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 計算兩個框的面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 計算 IoU
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value


def evaluate_colors(detections_list):
    # 計算多次評估的平均分數
    color_scores = {color: [] for color in expected_color_order}
    
    for detections in detections_list:
        for det in detections:
            if det['label'] in color_scores:
                color_scores[det['label']].append(det['confidence'])

    # 計算每個顏色的平均分數
    avg_scores = {color: (sum(scores) / len(scores) if scores else 0) for color, scores in color_scores.items()}

    # 判斷哪些顏色的平均分數超過閾值
    correct_colors = [color for color, score in avg_scores.items() if score >= confidence_threshold]

    return avg_scores, correct_colors

def check_color_order(correct_colors):
    # 根據已正確檢測的顏色，判斷顏色順序是否正確
    return correct_colors == expected_color_order

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

def run_inference(weights='yolov5s.pt', folder_path='', device='cpu', conf_thres=0.25, iou_thres=0.45, stop_threshold=0.7):
    model, stride, names, imgsz = load_model(weights, device, config_path)
    
    if not os.path.exists(folder_path):
        print(f"指定的資料夾 {folder_path} 不存在")
        return

    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            print(f"處理圖片: {image_path}")

            frame = cv2.imread(image_path)
            if frame is None:
                print(f"無法讀取圖片: {image_path}")
                continue

            # 多次評估特徵點，加入提前停止邏輯
            detections_list = []
            for i in range(evaluation_times):
                im, ratio, pad = preprocess_image(frame, device, imgsz)
                with torch.no_grad():
                    pred = model(im)
                _, detections = process_predictions(pred, im, frame, names, conf_thres, iou_thres, ratio, pad)
                detections_list.append(detections)

                # 計算顏色的平均分數
                avg_scores, _ = evaluate_colors(detections_list)
                print(f"圖片 {image_name} 在第 {i+1} 次檢測的平均分數: {avg_scores}")

                # 檢查是否所有顏色的平均分數都達到設定標準
                if all(score >= stop_threshold for score in avg_scores.values()):
                    print(f"所有顏色的平均分數都達到 {stop_threshold}，提前停止檢測。")
                    break  # 提前停止檢測

            # 最終計算顏色的平均分數並判斷是否正確
            avg_scores, correct_colors = evaluate_colors(detections_list)
            print(f"圖片 {image_name} 的最終平均分數: {avg_scores}")
            
            # 判斷顏色順序是否正確
            is_correct_order = check_color_order(correct_colors)
            if is_correct_order:
                print(f"圖片 {image_name} 的顏色順序正確")
            else:
                print(f"圖片 {image_name} 的顏色順序錯誤")

            # 繪製檢測框和標籤（僅繪製一次）
            result_frame, detections = process_predictions(pred, im, frame, names, conf_thres, iou_thres, ratio, pad)
            
            # 在左側固定順序顯示標籤
            draw_fixed_labels(result_frame, avg_scores)

            # 保存結果圖片
            output_path = os.path.join(folder_path, f"annotated_{image_name}")
            cv2.imwrite(output_path, result_frame)
            print(f"結果已保存到: {output_path}")



if __name__ == "__main__":
    folder_path = r'D:\Git\robotlearning\yoloface\target\result\data1'
    run_inference(weights='best.pt', folder_path=folder_path, device='cpu', conf_thres=0.5, iou_thres=0.6)
