# import cv2
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import json
# import os
# from typing import Dict, List, Tuple, Optional
# import logging

# class ColorPreprocessor:
#     @staticmethod
#     def enhance_image(image: np.ndarray) -> np.ndarray:
#         """增強圖像以提高檢測準確度"""
#         # 轉換到LAB空間進行光照均衡化
#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         cl = clahe.apply(l)
#         enhanced_lab = cv2.merge((cl,a,b))
#         return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
#     @staticmethod
#     def remove_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
#         """移除圖像噪點"""
#         return cv2.medianBlur(image, kernel_size)

# class ColorAnalyzer:
#     def __init__(self, base_path: str):
#         self.base_path = Path(base_path)
#         self.color_stats = {}
#         self.recommendations = {}
#         self.preprocessor = ColorPreprocessor()
        
#         # 設置日誌
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )
#         self.logger = logging.getLogger(__name__)

#     def load_images(self, color_name: str) -> List[np.ndarray]:
#         """載入並預處理指定顏色資料夾中的所有圖片"""
#         image_path = self.base_path / color_name
#         images = []
#         if image_path.exists():
#             for img_file in image_path.glob('*'):
#                 if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
#                     try:
#                         img = cv2.imread(str(img_file))
#                         if img is not None:
#                             # 預處理圖片
#                             img = self.preprocessor.enhance_image(img)
#                             img = self.preprocessor.remove_noise(img)
#                             images.append(img)
#                     except Exception as e:
#                         self.logger.error(f"處理圖片 {img_file} 時發生錯誤: {str(e)}")
#         return images

#     def analyze_color_space(self, images: List[np.ndarray], color_name: str):
#         """分析圖片在不同色彩空間的分布"""
#         if not images:
#             self.logger.warning(f"{color_name} 沒有有效的圖片樣本")
#             return

#         hsv_values = []
#         lab_values = []
        
#         for img in images:
#             # 計算主要顏色
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
#             # 使用K-means找出主要顏色
#             pixels = hsv.reshape(-1, 3)
#             kmeans = KMeans(n_clusters=1, n_init=10).fit(pixels)
#             hsv_dominant = kmeans.cluster_centers_[0]
            
#             pixels_lab = lab.reshape(-1, 3)
#             kmeans_lab = KMeans(n_clusters=1, n_init=10).fit(pixels_lab)
#             lab_dominant = kmeans_lab.cluster_centers_[0]
            
#             hsv_values.append(hsv_dominant)
#             lab_values.append(lab_dominant)
        
#         # 計算統計數據
#         self.color_stats[color_name] = {
#             'hsv': {
#                 'mean': np.mean(hsv_values, axis=0),
#                 'std': np.std(hsv_values, axis=0),
#                 'min': np.min(hsv_values, axis=0),
#                 'max': np.max(hsv_values, axis=0)
#             },
#             'lab': {
#                 'mean': np.mean(lab_values, axis=0),
#                 'std': np.std(lab_values, axis=0),
#                 'min': np.min(lab_values, axis=0),
#                 'max': np.max(lab_values, axis=0)
#             },
#             'sample_count': len(images)
#         }

#     def generate_recommendations(self):
#         """生成優化後的顏色檢測建議"""
#         for color_name, stats in self.color_stats.items():
#             hsv_stats = stats['hsv']
            
#             if color_name.lower() == 'red':
#                 # 紅色特殊處理：使用雙區間
#                 self.recommendations[color_name] = {
#                     'hsv_range': [
#                         {
#                             'lower': [0, 150, 50],
#                             'upper': [5, 255, 255]
#                         },
#                         {
#                             'lower': [170, 150, 50],
#                             'upper': [180, 255, 255]
#                         }
#                     ],
#                     'confidence': self._calculate_confidence(stats)
#                 }
#             else:
#                 # 其他顏色使用單區間
#                 mean = hsv_stats['mean']
#                 std = hsv_stats['std']
                
#                 # 根據標準差調整範圍，但確保不會太寬
#                 hsv_lower = np.maximum(mean - 1.5 * std, [0, 150, 50])
#                 hsv_upper = np.minimum(mean + 1.5 * std, [180, 255, 255])
                
#                 self.recommendations[color_name] = {
#                     'hsv_range': [{
#                         'lower': hsv_lower.tolist(),
#                         'upper': hsv_upper.tolist()
#                     }],
#                     'confidence': self._calculate_confidence(stats)
#                 }

#     def _calculate_confidence(self, stats: Dict) -> float:
#         """計算建議的可信度"""
#         sample_weight = min(stats['sample_count'] / 10, 1)
#         std_weight = np.mean(1 - stats['hsv']['std'] / [180, 255, 255])
#         return (sample_weight * 0.4 + std_weight * 0.6) * 100

#     def check_range_overlaps(self):
#         """檢查顏色範圍是否有重疊"""
#         overlaps = []
#         for color1 in self.recommendations:
#             for color2 in self.recommendations:
#                 if color1 >= color2:
#                     continue
                
#                 overlap = self._calculate_range_overlap(
#                     self.recommendations[color1],
#                     self.recommendations[color2]
#                 )
#                 if overlap > 0:
#                     overlaps.append({
#                         'colors': (color1, color2),
#                         'overlap_percentage': overlap
#                     })
#         return overlaps

#     def _calculate_range_overlap(self, range1: Dict, range2: Dict) -> float:
#         """計算兩個顏色範圍的重疊程度"""
#         def ranges_overlap(r1_lower, r1_upper, r2_lower, r2_upper):
#             return not (r1_upper[0] < r2_lower[0] or r2_upper[0] < r1_lower[0])
        
#         total_overlap = 0
#         # 處理可能的多區間情況
#         for r1 in range1['hsv_range']:
#             for r2 in range2['hsv_range']:
#                 if ranges_overlap(
#                     np.array(r1['lower']), 
#                     np.array(r1['upper']),
#                     np.array(r2['lower']), 
#                     np.array(r2['upper'])
#                 ):
#                     total_overlap += 1
                    
#         return (total_overlap / (len(range1['hsv_range']) * len(range2['hsv_range']))) * 100

# class ColorDetector:
#     def __init__(self, config_path: str):
#         """初始化顏色檢測器"""
#         with open(config_path, 'r') as f:
#             self.config = json.load(f)
#         self.preprocessor = ColorPreprocessor()

#     def detect_color(self, image: np.ndarray, color_name: str) -> Tuple[np.ndarray, float]:
#         """檢測指定顏色"""
#         if color_name not in self.config:
#             raise ValueError(f"未知的顏色: {color_name}")
            
#         # 預處理
#         processed = self.preprocessor.enhance_image(image)
#         processed = self.preprocessor.remove_noise(processed)
        
#         # 轉換到HSV
#         hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
#         # 合併所有區間的遮罩
#         final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         for range_config in self.config[color_name]['hsv_range']:
#             lower = np.array(range_config['lower'])
#             upper = np.array(range_config['upper'])
#             mask = cv2.inRange(hsv, lower, upper)
#             final_mask = cv2.bitwise_or(final_mask, mask)
            
#         # 計算檢測置信度
#         confidence = (np.sum(final_mask > 0) / final_mask.size) * 100
            
#         return final_mask, confidence

# def main():
#     # 設置路徑
#     base_path = "./color"
#     output_path = "./analysis_results"
#     Path(output_path).mkdir(parents=True, exist_ok=True)
    
#     # 1. 分析顏色樣本
#     analyzer = ColorAnalyzer(base_path)
#     print("開始分析顏色樣本...")
    
#     # 分析每個顏色資料夾
#     for color_folder in analyzer.base_path.iterdir():
#         if color_folder.is_dir():
#             color_name = color_folder.name
#             print(f"分析 {color_name}...")
#             images = analyzer.load_images(color_name)
#             if images:
#                 analyzer.analyze_color_space(images, color_name)
                
#     # 生成建議
#     analyzer.generate_recommendations()
    
#     # 檢查重疊
#     overlaps = analyzer.check_range_overlaps()
#     if overlaps:
#         print("\n發現顏色範圍重疊:")
#         for overlap in overlaps:
#             print(f"{overlap['colors'][0]} 和 {overlap['colors'][1]} 重疊 {overlap['overlap_percentage']:.2f}%")
    
#     # 保存設置
#     config_path = Path(output_path) / 'color_config.json'
#     with open(config_path, 'w', encoding='utf-8') as f:
#         json.dump(analyzer.recommendations, f, indent=2, ensure_ascii=False)
    
#     print(f"\n配置已保存到 {config_path}")
    
#     # 2. 測試顏色檢測
#     detector = ColorDetector(str(config_path))
    
#     # 測試每個顏色資料夾中的圖片
#     for color_folder in Path(base_path).iterdir():
#         if color_folder.is_dir():
#             color_name = color_folder.name
#             print(f"\n測試 {color_name} 檢測...")
            
#             test_images = [cv2.imread(str(f)) for f in color_folder.glob('*')
#                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
#             if test_images:
#                 success_count = 0
#                 for img in test_images:
#                     mask, confidence = detector.detect_color(img, color_name)
#                     if confidence > 30:  # 可調整的閾值
#                         success_count += 1
                
#                 success_rate = (success_count / len(test_images)) * 100
#                 print(f"檢測成功率: {success_rate:.2f}%")

# if __name__ == "__main__":
#     main()



import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

class ColorPreprocessor:
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """增強圖像以提高檢測準確度"""
        # 轉換到LAB空間進行光照均衡化
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """移除圖像噪點"""
        return cv2.medianBlur(image, kernel_size)

class ColorAnalyzer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.color_stats = {}
        self.recommendations = {}
        self.preprocessor = ColorPreprocessor()
        
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_images(self, color_name: str) -> List[np.ndarray]:
        """載入並預處理指定顏色資料夾中的所有圖片"""
        image_path = self.base_path / color_name
        images = []
        if image_path.exists():
            for img_file in image_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            # 預處理圖片
                            img = self.preprocessor.enhance_image(img)
                            img = self.preprocessor.remove_noise(img)
                            images.append(img)
                    except Exception as e:
                        self.logger.error(f"處理圖片 {img_file} 時發生錯誤: {str(e)}")
        return images

    def analyze_color_space(self, images: List[np.ndarray], color_name: str):
        """分析圖片在不同色彩空間的分布"""
        if not images:
            self.logger.warning(f"{color_name} 沒有有效的圖片樣本")
            return

        hsv_values = []
        lab_values = []
        
        for img in images:
            # 計算主要顏色
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # 使用K-means找出主要顏色
            pixels = hsv.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, n_init=10).fit(pixels)
            hsv_dominant = kmeans.cluster_centers_[0]
            
            pixels_lab = lab.reshape(-1, 3)
            kmeans_lab = KMeans(n_clusters=1, n_init=10).fit(pixels_lab)
            lab_dominant = kmeans_lab.cluster_centers_[0]
            
            hsv_values.append(hsv_dominant)
            lab_values.append(lab_dominant)
        
        # 計算統計數據
        self.color_stats[color_name] = {
            'hsv': {
                'mean': np.mean(hsv_values, axis=0),
                'std': np.std(hsv_values, axis=0),
                'min': np.min(hsv_values, axis=0),
                'max': np.max(hsv_values, axis=0)
            },
            'lab': {
                'mean': np.mean(lab_values, axis=0),
                'std': np.std(lab_values, axis=0),
                'min': np.min(lab_values, axis=0),
                'max': np.max(lab_values, axis=0)
            },
            'sample_count': len(images)
        }

    def generate_recommendations(self):
        """生成優化後的顏色檢測建議"""
        for color_name, stats in self.color_stats.items():
            hsv_stats = stats['hsv']
            lab_stats = stats['lab']
            
            if color_name.lower() == 'red':
                # 紅色特殊處理：使用雙區間
                self.recommendations[color_name] = {
                    'hsv_range': [
                        {
                            'lower': [0, 150, 50],
                            'upper': [5, 255, 255]
                        },
                        {
                            'lower': [170, 150, 50],
                            'upper': [180, 255, 255]
                        }
                    ],
                    'lab_range': {
                        'lower': (lab_stats['mean'] - 1.5 * lab_stats['std']).tolist(),
                        'upper': (lab_stats['mean'] + 1.5 * lab_stats['std']).tolist()
                    },
                    'confidence': self._calculate_confidence(stats)
                }
            else:
                # 其他顏色使用單區間
                mean_hsv = hsv_stats['mean']
                std_hsv = hsv_stats['std']
                mean_lab = lab_stats['mean']
                std_lab = lab_stats['std']
                
                # HSV 範圍
                hsv_lower = np.maximum(mean_hsv - 1.5 * std_hsv, [0, 150, 50])
                hsv_upper = np.minimum(mean_hsv + 1.5 * std_hsv, [180, 255, 255])
                
                # LAB 範圍
                lab_lower = np.maximum(mean_lab - 1.5 * std_lab, [0, 0, 0])
                lab_upper = np.minimum(mean_lab + 1.5 * std_lab, [255, 255, 255])
                
                self.recommendations[color_name] = {
                    'hsv_range': [{
                        'lower': hsv_lower.tolist(),
                        'upper': hsv_upper.tolist()
                    }],
                    'lab_range': {
                        'lower': lab_lower.tolist(),
                        'upper': lab_upper.tolist()
                    },
                    'confidence': self._calculate_confidence(stats)
                }

    def _calculate_confidence(self, stats: Dict) -> float:
        """計算建議的可信度"""
        # 考慮兩個色彩空間的樣本權重
        sample_weight = min(stats['sample_count'] / 10, 1)
        
        # HSV和LAB的標準差權重
        hsv_std_weight = np.mean(1 - stats['hsv']['std'] / [180, 255, 255])
        lab_std_weight = np.mean(1 - stats['lab']['std'] / [255, 255, 255])
        
        # 綜合計算
        return (sample_weight * 0.3 + hsv_std_weight * 0.4 + lab_std_weight * 0.3) * 100

    def check_range_overlaps(self):
        """檢查顏色範圍是否有重疊"""
        overlaps = []
        for color1 in self.recommendations:
            for color2 in self.recommendations:
                if color1 >= color2:
                    continue
                
                overlap = self._calculate_range_overlap(
                    self.recommendations[color1],
                    self.recommendations[color2]
                )
                if overlap > 0:
                    overlaps.append({
                        'colors': (color1, color2),
                        'overlap_percentage': overlap
                    })
        return overlaps

    def _calculate_range_overlap(self, range1: Dict, range2: Dict) -> float:
        """計算兩個顏色範圍的重疊程度"""
        def ranges_overlap(r1_lower, r1_upper, r2_lower, r2_upper):
            return not (r1_upper[0] < r2_lower[0] or r2_upper[0] < r1_lower[0])
        
        total_overlap = 0
        # 處理HSV重疊
        for r1 in range1['hsv_range']:
            for r2 in range2['hsv_range']:
                if ranges_overlap(
                    np.array(r1['lower']), 
                    np.array(r1['upper']),
                    np.array(r2['lower']), 
                    np.array(r2['upper'])
                ):
                    total_overlap += 0.5  # HSV重疊權重
                    
        # 處理LAB重疊
        if ranges_overlap(
            np.array(range1['lab_range']['lower']),
            np.array(range1['lab_range']['upper']),
            np.array(range2['lab_range']['lower']),
            np.array(range2['lab_range']['upper'])
        ):
            total_overlap += 0.5  # LAB重疊權重
                    
        return total_overlap * 100

class ColorDetector:
    def __init__(self, config_path: str):
        """初始化顏色檢測器"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.preprocessor = ColorPreprocessor()

    def detect_color(self, image: np.ndarray, color_name: str) -> Tuple[np.ndarray, float]:
        """檢測指定顏色"""
        if color_name not in self.config:
            raise ValueError(f"未知的顏色: {color_name}")
            
        # 預處理
        processed = self.preprocessor.enhance_image(image)
        processed = self.preprocessor.remove_noise(processed)
        
        # HSV 檢測
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        hsv_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for range_config in self.config[color_name]['hsv_range']:
            lower = np.array(range_config['lower'])
            upper = np.array(range_config['upper'])
            mask = cv2.inRange(hsv, lower, upper)
            hsv_mask = cv2.bitwise_or(hsv_mask, mask)
        
        # LAB 檢測
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        lab_lower = np.array(self.config[color_name]['lab_range']['lower'])
        lab_upper = np.array(self.config[color_name]['lab_range']['upper'])
        lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
        
        # 結合 HSV 和 LAB 的結果
        final_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        
        # 使用形態學操作去除噪點
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # 計算綜合置信度
        hsv_confidence = (np.sum(hsv_mask > 0) / hsv_mask.size) * 100
        lab_confidence = (np.sum(lab_mask > 0) / lab_mask.size) * 100
        final_confidence = (np.sum(final_mask > 0) / final_mask.size) * 100
        
        # 綜合置信度計算
        weighted_confidence = (hsv_confidence * 0.6 + lab_confidence * 0.4 + final_confidence) / 3
            
        return final_mask, weighted_confidence

    def get_confidence_details(self, image: np.ndarray, color_name: str) -> Dict:
        """獲取詳細的置信度資訊"""
        processed = self.preprocessor.enhance_image(image)
        processed = self.preprocessor.remove_noise(processed)
        
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        
        # HSV 檢測
        hsv_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for range_config in self.config[color_name]['hsv_range']:
            lower = np.array(range_config['lower'])
            upper = np.array(range_config['upper'])
            mask = cv2.inRange(hsv, lower, upper)
            hsv_mask = cv2.bitwise_or(hsv_mask, mask)
        
        # LAB 檢測
        lab_lower = np.array(self.config[color_name]['lab_range']['lower'])
        lab_upper = np.array(self.config[color_name]['lab_range']['upper'])
        lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
        
        # 結合結果
        final_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        
        # 形態學處理
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return {
            'hsv_confidence': (np.sum(hsv_mask > 0) / hsv_mask.size) * 100,
            'lab_confidence': (np.sum(lab_mask > 0) / lab_mask.size) * 100,
            'final_confidence': (np.sum(final_mask > 0) / final_mask.size) * 100,
            'hsv_mask': hsv_mask,
            'lab_mask': lab_mask,
            'final_mask': final_mask
        }

def visualize_results(image: np.ndarray, detection_results: Dict, save_path: Optional[str] = None):
    """視覺化檢測結果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原圖
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始圖片')
    axes[0, 0].axis('off')
    
    # HSV 遮罩
    axes[0, 1].imshow(detection_results['hsv_mask'], cmap='gray')
    axes[0, 1].set_title(f'HSV 遮罩\n置信度: {detection_results["hsv_confidence"]:.2f}%')
    axes[0, 1].axis('off')
    
    # LAB 遮罩
    axes[1, 0].imshow(detection_results['lab_mask'], cmap='gray')
    axes[1, 0].set_title(f'LAB 遮罩\n置信度: {detection_results["lab_confidence"]:.2f}%')
    axes[1, 0].axis('off')
    
    # 最終遮罩
    axes[1, 1].imshow(detection_results['final_mask'], cmap='gray')
    axes[1, 1].set_title(f'最終遮罩\n置信度: {detection_results["final_confidence"]:.2f}%')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # 設置路徑
    base_path = "./color"
    output_path = "./analysis_results"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 1. 分析顏色樣本
    analyzer = ColorAnalyzer(base_path)
    print("開始分析顏色樣本...")
    
    # 分析每個顏色資料夾
    for color_folder in analyzer.base_path.iterdir():
        if color_folder.is_dir():
            color_name = color_folder.name
            print(f"分析 {color_name}...")
            images = analyzer.load_images(color_name)
            if images:
                analyzer.analyze_color_space(images, color_name)
                
    # 生成建議
    analyzer.generate_recommendations()
    
    # 檢查重疊
    overlaps = analyzer.check_range_overlaps()
    if overlaps:
        print("\n發現顏色範圍重疊:")
        for overlap in overlaps:
            print(f"{overlap['colors'][0]} 和 {overlap['colors'][1]} 重疊 {overlap['overlap_percentage']:.2f}%")
    
    # 保存設置
    config_path = Path(output_path) / 'color_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(analyzer.recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存到 {config_path}")
    
    # 2. 測試顏色檢測
    detector = ColorDetector(str(config_path))
    results_path = Path(output_path) / 'test_results'
    results_path.mkdir(exist_ok=True)
    
    # 測試每個顏色資料夾中的圖片
    for color_folder in Path(base_path).iterdir():
        if color_folder.is_dir():
            color_name = color_folder.name
            print(f"\n測試 {color_name} 檢測...")
            
            test_files = list(color_folder.glob('*'))
            test_files = [f for f in test_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if test_files:
                success_count = 0
                for i, img_path in enumerate(test_files):
                    # 讀取圖片
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # 獲取詳細檢測結果
                    detection_details = detector.get_confidence_details(img, color_name)
                    
                    # 儲存視覺化結果
                    save_path = results_path / f'{color_name}_test_{i}.png'
                    visualize_results(img, detection_details, str(save_path))
                    
                    # 判斷是否檢測成功
                    if detection_details['final_confidence'] > 30:  # 可調整的閾值
                        success_count += 1
                
                success_rate = (success_count / len(test_files)) * 100
                print(f"檢測成功率: {success_rate:.2f}%")
                print(f"測試結果已保存到: {results_path}")

if __name__ == "__main__":
    main()