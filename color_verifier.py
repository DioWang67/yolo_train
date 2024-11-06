# color_verifier.py
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
            ],
            'Green': [
                {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])}
            ]
        }

    def find_wire_mask(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            area = cv2.contourArea(contour)
            if (0.1 < aspect_ratio < 10) and area > 100:
                cv2.drawContours(mask, [contour], -1, (255), -1)
        
        return mask

    def verify_color(self, frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0
        
        wire_mask = self.find_wire_mask(roi)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_scores = {}
        for color_name, ranges in self.color_ranges.items():
            total_score = 0
            for range_dict in ranges:
                color_mask = cv2.inRange(hsv_roi, range_dict['lower'], range_dict['upper'])
                combined_mask = cv2.bitwise_and(color_mask, color_mask, mask=wire_mask)
                wire_pixels = np.sum(wire_mask == 255)
                if wire_pixels > 0:
                    score = np.sum(combined_mask == 255) / wire_pixels
                    total_score += score
            color_scores[color_name] = total_score
        
        verified_color = max(color_scores.items(), key=lambda x: x[1])
        
        if verified_color[1] < 0.15:
            return None, 0
            
        return verified_color