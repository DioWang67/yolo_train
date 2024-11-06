import cv2
import numpy as np

class ColorVerifier:
    def __init__(self):
        # 定義 HSV 顏色範圍
        self.color_ranges = {
            'Red': [
                {'lower': np.array([0, 120, 50]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([160, 120, 50]), 'upper': np.array([180, 255, 255])}
            ],
            'Orange': [
                {'lower': np.array([5, 150, 50]), 'upper': np.array([25, 255, 255])}
            ],
            'Yellow': [
                {'lower': np.array([20, 100, 50]), 'upper': np.array([35, 255, 255])}
            ],
            'Black': [
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])}
            ]
        }

    def find_wire_mask(self, roi):
        """使用邊緣檢測和形態學運算來找出電線區域"""
        # 轉換為灰度圖
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊減少噪聲
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny邊緣檢測
        edges = cv2.Canny(blur, 50, 150)
        
        # 膨脹操作以連接邊緣
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 尋找連通區域
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 創建遮罩
        mask = np.zeros_like(gray)
        
        # 篩選細長的連通區域（可能是電線）
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            area = cv2.contourArea(contour)
            
            # 根據面積和長寬比篩選
            if (0.1 < aspect_ratio < 10) and area > 100:  # 這些閾值可以調整
                cv2.drawContours(mask, [contour], -1, (255), -1)
        
        return mask

    def verify_color(self, frame, box, predicted_label, confidence):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return predicted_label, confidence
        
        # 找出電線區域
        wire_mask = self.find_wire_mask(roi)
        
        # 轉換到 HSV 顏色空間
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 計算每種顏色在電線區域的得分
        color_scores = {}
        for color_name, ranges in self.color_ranges.items():
            total_score = 0
            for range_dict in ranges:
                # 創建顏色遮罩
                color_mask = cv2.inRange(hsv_roi, range_dict['lower'], range_dict['upper'])
                # 將顏色遮罩與電線遮罩結合
                combined_mask = cv2.bitwise_and(color_mask, color_mask, mask=wire_mask)
                # 計算分數
                wire_pixels = np.sum(wire_mask == 255)
                if wire_pixels > 0:  # 避免除以零
                    score = np.sum(combined_mask == 255) / wire_pixels
                    total_score += score
            color_scores[color_name] = total_score
        
        # 獲取最高分的顏色
        verified_color = max(color_scores.items(), key=lambda x: x[1])
        
        # 如果分數太低，保持原始預測
        if verified_color[1] < 0.15:  # 可調整的閾值
            return predicted_label, confidence
            
        return verified_color[0], verified_color[1]

def draw_debug_visualization(frame, box, wire_mask, color_masks):
    """繪製偵錯視覺化（可選）"""
    x1, y1, x2, y2 = box
    debug_frame = frame.copy()
    
    # 顯示電線遮罩
    debug_frame[y1:y2, x1:x2][wire_mask > 0] = [0, 255, 0]
    
    # 顯示每種顏色的遮罩
    for color_name, mask in color_masks.items():
        color = {'Red': (0,0,255), 'Orange': (0,165,255),
                'Yellow': (0,255,255), 'Black': (128,128,128)}[color_name]
        debug_frame[y1:y2, x1:x2][mask > 0] = color
    
    return debug_frame