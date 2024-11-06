import cv2
from yolov5_inference import run_inference

def main():
    # 設置參數
    source = 0  # 使用攝像頭
    weights = 'yolov5s.pt'
    device = 'cpu'
    
    # 運行攝像頭檢測
    results = run_inference(weights=weights, source=source, device=device)

    # 顯示攝像頭檢測結果
    for result in results:
        cv2.imshow('YOLOv5 Camera Detection', result['image'])

        # 如果按下 'q' 鍵，退出檢測
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
