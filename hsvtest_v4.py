import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# 設定中文字型和負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

class ColorPreprocessor:
    """影像預處理類別"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """
        增強影像以提高偵測準確度
        
        參數:
            image: 輸入影像
        回傳:
            處理後的影像
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        移除影像雜訊
        
        參數:
            image: 輸入影像
            kernel_size: 濾波核心大小
        回傳:
            處理後的影像
        """
        return cv2.medianBlur(image, kernel_size)

class ColorAnalyzer:
    """顏色分析類別"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.color_stats = {}
        self.recommendations = {}
        self.preprocessor = ColorPreprocessor()
        
        # 設定記錄檔
        log_path = self.base_path / 'logs'
        log_path.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / f'color_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
                                    encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_images(self, color_name: str) -> List[np.ndarray]:
        """載入並預處理指定顏色資料夾中的所有影像"""
        image_path = self.base_path / color_name
        images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        if not image_path.exists():
            self.logger.warning(f"找不到顏色資料夾：{color_name}")
            return images
            
        for img_file in image_path.glob('*'):
            if img_file.suffix.lower() in valid_extensions:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        self.logger.warning(f"無法讀取影像：{img_file}")
                        continue
                        
                    # 預處理影像
                    img = self.preprocessor.enhance_image(img)
                    img = self.preprocessor.remove_noise(img)
                    images.append(img)
                    self.logger.info(f"成功載入影像：{img_file}")
                except Exception as e:
                    self.logger.error(f"處理影像 {img_file} 時發生錯誤：{str(e)}")
        
        return images

    def analyze_color_space(self, images: List[np.ndarray], color_name: str):
        """分析影像在不同色彩空間的分布"""
        if not images:
            self.logger.warning(f"{color_name} 沒有有效的影像樣本")
            return

        hsv_values = []
        lab_values = []
        
        for idx, img in enumerate(images):
            try:
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
                
                self.logger.debug(f"已分析 {color_name} 的第 {idx+1} 張影像")
            except Exception as e:
                self.logger.error(f"分析 {color_name} 的第 {idx+1} 張影像時發生錯誤：{str(e)}")
        
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
        
        self.logger.info(f"完成 {color_name} 的顏色空間分析，共處理 {len(images)} 張影像")

    def generate_recommendations(self, std_multiplier: float = 1.0):
        """產生優化後的顏色偵測建議值"""
        for color_name, stats in self.color_stats.items():
            try:
                hsv_stats = stats['hsv']
                lab_stats = stats['lab']

                if color_name.lower() == 'red':
                    # 紅色特殊處理：使用雙區間
                    mean_lab = lab_stats['mean']
                    std_lab = lab_stats['std']
                    lab_lower = np.maximum(mean_lab - std_multiplier * std_lab, [0, 0, 0])
                    lab_upper = np.minimum(mean_lab + std_multiplier * std_lab, [255, 255, 255])

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
                            'lower': lab_lower.tolist(),
                            'upper': lab_upper.tolist()
                        },
                        'confidence': self._calculate_confidence(stats)
                    }
                else:
                    # 其他顏色使用單區間
                    mean_hsv = hsv_stats['mean']
                    std_hsv = hsv_stats['std']
                    mean_lab = lab_stats['mean']
                    std_lab = lab_stats['std']

                    hsv_lower = np.maximum(mean_hsv - std_multiplier * std_hsv, [0, 150, 50])
                    hsv_upper = np.minimum(mean_hsv + std_multiplier * std_hsv, [180, 255, 255])

                    lab_lower = np.maximum(mean_lab - std_multiplier * std_lab, [0, 0, 0])
                    lab_upper = np.minimum(mean_lab + std_multiplier * std_lab, [255, 255, 255])

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
                
                self.logger.info(f"已生成 {color_name} 的建議值")
            except Exception as e:
                self.logger.error(f"生成 {color_name} 的建議值時發生錯誤：{str(e)}")

    def _calculate_confidence(self, stats: Dict) -> float:
        """計算建議值的可信度"""
        try:
            sample_weight = min(stats['sample_count'] / 10, 1)
            hsv_std_weight = np.mean(1 - stats['hsv']['std'] / [180, 255, 255])
            lab_std_weight = np.mean(1 - stats['lab']['std'] / [255, 255, 255])
            return (sample_weight * 0.3 + hsv_std_weight * 0.4 + lab_std_weight * 0.3) * 100
        except Exception as e:
            self.logger.error(f"計算可信度時發生錯誤：{str(e)}")
            return 0.0

    def visualize_color_distribution(self, output_path: str):
        """視覺化顏色分布"""
        for color_name, stats in self.color_stats.items():
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # HSV 分布
                hsv_mean = stats['hsv']['mean']
                hsv_std = stats['hsv']['std']
                ax1.bar(['H', 'S', 'V'], hsv_mean)
                ax1.errorbar(['H', 'S', 'V'], hsv_mean, yerr=hsv_std, fmt='none', color='black')
                ax1.set_title(f'{color_name} - HSV 分布')
                
                # LAB 分布
                lab_mean = stats['lab']['mean']
                lab_std = stats['lab']['std']
                ax2.bar(['L', 'a', 'b'], lab_mean)
                ax2.errorbar(['L', 'a', 'b'], lab_mean, yerr=lab_std, fmt='none', color='black')
                ax2.set_title(f'{color_name} - LAB 分布')
                
                plt.tight_layout()
                plt.savefig(Path(output_path) / f'{color_name}_distribution.png')
                plt.close()
                
                self.logger.info(f"已儲存 {color_name} 的顏色分布圖")
            except Exception as e:
                self.logger.error(f"視覺化 {color_name} 的顏色分布時發生錯誤：{str(e)}")


class ColorDetector:
    """改進的顏色檢測器"""
    
    def __init__(self, config_path: str):
        """初始化顏色檢測器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.preprocessor = ColorPreprocessor()
        
        # 新增顏色區域最小面積閾值（可調整）
        self.min_area_threshold = 100
        # 新增雜訊過濾參數
        self.noise_kernel_size = 3
        self.morph_kernel_size = 5
        
    def detect_color(self, image: np.ndarray, color_name: str) -> Tuple[np.ndarray, float, Dict]:
        """
        改進的顏色檢測方法
        
        參數:
            image: 輸入影像
            color_name: 要檢測的顏色名稱
            
        回傳:
            遮罩, 置信度, 詳細資訊
        """
        if color_name not in self.config:
            raise ValueError(f"未知的顏色: {color_name}")
        
        # 預處理
        processed = self._preprocess_image(image)
        
        # 多階段檢測
        hsv_results = self._detect_hsv(processed, color_name)
        lab_results = self._detect_lab(processed, color_name)
        
        # 結合結果並進行後處理
        final_mask, confidence, details = self._combine_results(hsv_results, lab_results)
        
        return final_mask, confidence, details
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """改進的影像預處理"""
        # 基本預處理
        processed = self.preprocessor.enhance_image(image)
        processed = self.preprocessor.remove_noise(processed)
        
        # 額外的預處理步驟: 高斯模糊+亮度對比度調整
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return processed
    
    def _detect_hsv(self, image: np.ndarray, color_name: str) -> Dict:
        """改進的HSV空間檢測"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 收集每個區間的遮罩
        individual_masks = []
        for range_config in self.config[color_name]['hsv_range']:
            lower = np.array(range_config['lower'])
            upper = np.array(range_config['upper'])
            
            # 套用遮罩
            mask = cv2.inRange(hsv, lower, upper)
            individual_masks.append(mask)
            hsv_mask = cv2.bitwise_or(hsv_mask, mask)
        
        # 移除小區域
        hsv_mask = self._remove_small_regions(hsv_mask)
        
        # 計算HSV空間的特徵
        hsv_features = self._calculate_color_features(hsv, hsv_mask)
        
        return {
            'mask': hsv_mask,
            'features': hsv_features,
            'individual_masks': individual_masks
        }
    
    def _detect_lab(self, image: np.ndarray, color_name: str) -> Dict:
        """改進的LAB空間檢測"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        lab_config = self.config[color_name]['lab_range']
        lab_lower = np.array(lab_config['lower'])
        lab_upper = np.array(lab_config['upper'])
        
        # 套用遮罩
        lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
        
        # 移除小區域
        lab_mask = self._remove_small_regions(lab_mask)
        
        # 計算LAB空間的特徵
        lab_features = self._calculate_color_features(lab, lab_mask)
        
        return {
            'mask': lab_mask,
            'features': lab_features
        }
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """移除小區域"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 從1開始以跳過背景
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_threshold:
                cleaned_mask[labels == i] = 255
                
        return cleaned_mask
    
    def _calculate_color_features(self, color_space_image: np.ndarray, mask: np.ndarray) -> Dict:
        """計算顏色特徵"""
        if np.sum(mask) == 0:
            return {
                'mean_values': np.zeros(3),
                'std_values': np.zeros(3),
                'coverage': 0.0
            }
        
        masked_pixels = color_space_image[mask > 0]
        
        return {
            'mean_values': np.mean(masked_pixels, axis=0),
            'std_values': np.std(masked_pixels, axis=0),
            'coverage': np.sum(mask > 0) / mask.size
        }
    
    def _combine_results(self, hsv_results: Dict, lab_results: Dict) -> Tuple[np.ndarray, float, Dict]:
        """智慧型結合HSV和LAB結果"""
        hsv_mask = hsv_results['mask']
        lab_mask = lab_results['mask']
        
        # 結合遮罩
        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        
        # 形態學處理
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        hsv_confidence = hsv_results['features']['coverage'] * 100
        lab_confidence = lab_results['features']['coverage'] * 100
        
        # 動態調整權重
        hsv_weight = 0.6
        lab_weight = 0.4
        if hsv_confidence > lab_confidence * 1.5:
            hsv_weight = 0.8
            lab_weight = 0.2
        elif lab_confidence > hsv_confidence * 1.5:
            hsv_weight = 0.2
            lab_weight = 0.8
        
        final_confidence = (hsv_confidence * hsv_weight + lab_confidence * lab_weight +
                            (np.sum(combined_mask > 0) / combined_mask.size * 100)) / 3
        
        details = {
            'hsv_confidence': hsv_confidence,
            'lab_confidence': lab_confidence,
            'hsv_features': hsv_results['features'],
            'lab_features': lab_results['features'],
            'weights': {
                'hsv': hsv_weight,
                'lab': lab_weight
            }
        }
        
        return combined_mask, final_confidence, details


def detect_and_visualize(image_path: str, color_name: str, detector: ColorDetector):
    """檢測並視覺化結果"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取影像：{image_path}")
        return
    
    # 執行檢測
    mask, confidence, details = detector.detect_color(image, color_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始影像
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始影像')
    axes[0, 0].axis('off')
    
    # 遮罩結果
    axes[0, 1].imshow(cv2.bitwise_and(image, image, mask=mask))
    axes[0, 1].set_title(f'遮罩結果\n置信度: {confidence:.2f}%')
    axes[0, 1].axis('off')
    
    # 顯示HSV和LAB的置信度
    axes[1, 0].bar(['HSV', 'LAB'], [details['hsv_confidence'], details['lab_confidence']])
    axes[1, 0].set_title('色彩空間置信度')
    
    # 權重分布
    axes[1, 1].pie([details['weights']['hsv'], details['weights']['lab']], 
                   labels=['HSV', 'LAB'],
                   autopct='%1.1f%%')
    axes[1, 1].set_title('權重分布')
    
    plt.tight_layout()
    plt.show()


def main():
    # 設定路徑
    base_path = "./color"
    output_path = "./analysis_results"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 初始化分析器
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
    
    # 生成建議值和視覺化結果
    analyzer.generate_recommendations()
    analyzer.visualize_color_distribution(output_path)
    
    # 儲存設定
    config_path = Path(output_path) / 'color_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(analyzer.recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"分析完成！結果已儲存至：{output_path}")
    
    # 若要測試偵測結果，可使用下方範例 (需自行提供測試影像與顏色名稱)
    # detector = ColorDetector(str(config_path))
    # detect_and_visualize('./test_image.jpg', 'red', detector)

if __name__ == "__main__":
    main()
