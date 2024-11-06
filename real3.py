from ultralytics import YOLO
import cv2
import time
from collections import defaultdict

def main():
    # 加载您训练好的 YOLOv8 模型
    model = YOLO('best1.pt')  # 请确保模型文件路径正确

    # 打开 USB 摄像头（将索引替换为您的摄像头索引）
    camera_index = 1  # USB 摄像头的索引
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"无法打开摄像头索引 {camera_index}")
        return

    scan_duration = 60  # 扫描时长（秒）
    start_time = time.time()  # 开始时间

    # 期望的线色顺序（从下到上：红、绿、橙、黄、黑、黑）
    expected_sequence = ['bla', 'bla', 'yel', 'org', 'gre', 'red']
    positions = [None] * len(expected_sequence)  # 初始化6个箱子

    # 存储每种颜色的置信度和检测次数
    confidence_sums = defaultdict(float)  # 置信度累加器
    detection_counts = defaultdict(int)    # 检测次数累加器

    prev_time = time.time()

    # 标志位，用于停止摄像头扫描
    stop_scanning = False

    while True:
        # 在5秒内进行检测
        if not stop_scanning and time.time() - start_time <= scan_duration:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头数据")
                break

            height, width, _ = frame.shape

            # 使用 YOLOv8 进行物体检测
            results = model(frame)

            # 获取带有检测结果的图像
            annotated_frame = results[0].plot()

            # 处理检测结果，优先找到 p+
            boxes = results[0].boxes
            p_plus_position = None
            color_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = box.conf[0].item()  # 获取置信度

                # 累积置信度和检测次数
                confidence_sums[class_name] += confidence
                detection_counts[class_name] += 1

                if class_name == 'p+':
                    # 找到 p+ 的位置，存储它的坐标
                    p_plus_position = (x1, y1, x2, y2)
                else:
                    # 记录所有线色类别的框
                    color_boxes.append((x1, x2, y1, y2, class_name))

            # 如果找到了 p+，开始从右边到上方检测颜色
            if p_plus_position:
                p_x1, p_y1, p_x2, p_y2 = p_plus_position

                # 从 p+ 的右边开始，逐步向上检测颜色
                for x1, x2, y1, y2, class_name in sorted(color_boxes, key=lambda x: -x[2]):
                    if (x1 > p_x2 - 10) and (y2 < p_y2 + 10):  # 放宽条件
                        # 计算平均置信度
                        avg_confidence = confidence_sums[class_name] / detection_counts[class_name]

                        if avg_confidence < 0.3:
                            # 如果平均置信度小于0.5，跳过该颜色
                            continue

                        if class_name == 'bla':  # 处理黑色
                            # 找到黑色在 `expected_sequence` 中的所有位置
                            black_indices = [i for i, color in enumerate(expected_sequence) if color == 'bla']
                            for index in black_indices:
                                if positions[index] is None:  # 确保每个位置都能记录黑色
                                    positions[index] = class_name
                                    break  # 记录完一个黑色后，继续检测其他颜色
                        else:
                            if class_name in expected_sequence:
                                index = expected_sequence.index(class_name)
                                if positions[index] is None:
                                    positions[index] = class_name
                            else:
                                print(f"颜色 {class_name} 已经记录或者不在预期中")
                    if all(positions):  # 如果已经找到 6 个颜色，结束检测
                        break

            # 在图像上显示当前检测到的线色
            start_x = 10
            start_y = 50
            line_height = 30
            for i, class_name in enumerate(positions):
                display_text = f"Pos {i+1}: {class_name if class_name else 'x'}"
                y_position = start_y + i * line_height
                cv2.putText(annotated_frame, display_text, (start_x, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 计算成功率
            matched_positions = sum(1 for a, e in zip(positions, expected_sequence) if a == e)
            total_positions = len(expected_sequence)
            success_rate = (matched_positions / total_positions) * 100

            # 5秒后停止扫描
            if time.time() - start_time >= scan_duration:
                stop_scanning = True  # 停止扫描

                # 判断PASS或FAIL
                if success_rate == 100:
                    result_text = "PASS"
                    color = (0, 255, 0)  # 绿色
                else:
                    result_text = "FAIL"
                    color = (0, 0, 255)  # 红色

                # 在图像中央显示结果
                cv2.putText(annotated_frame, result_text, (width // 2 - 50, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        # 显示最后的结果
        cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 输出累积的线色顺序
    print("\n扫描结果：")
    print(f"累积的线色顺序：{positions}")
    print(f"期望的线色顺序：{expected_sequence}")

    # 比较累积的线色顺序与期望顺序
    success = positions == expected_sequence
    print(f"成功匹配：{'是' if success else '否'}")

    print(f"成功率：{success_rate:.2f}%")

if __name__ == "__main__":
    main()