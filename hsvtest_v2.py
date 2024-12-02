class DynamicHSVRange:
    def __init__(self, initial_ranges):
        """
        初始化範圍。
        :param initial_ranges: 初始範圍列表，每個範圍為 [[下界], [上界]]
        """
        self.ranges = initial_ranges

    def adjust_range(self, hsv_value):
        """
        調整範圍，使HSV值包含在內。
        :param hsv_value: 要包含的HSV值 (H, S, V)
        """
        for lower_bound, upper_bound in self.ranges:
            for i in range(3):  # H, S, V 各自調整
                lower_bound[i] = min(lower_bound[i], hsv_value[i])
                upper_bound[i] = max(upper_bound[i], hsv_value[i])

    def is_in_any_range(self, hsv_value):
        """
        檢查HSV值是否在任意範圍內。
        :param hsv_value: 要檢查的HSV值 (H, S, V)
        :return: 如果在任意範圍內返回 True，否則返回 False
        """
        for lower_bound, upper_bound in self.ranges:
            if all(lower_bound[i] <= hsv_value[i] <= upper_bound[i] for i in range(3)):
                return True
        return False

    def get_current_ranges(self):
        """
        獲取當前所有範圍。
        :return: 當前範圍列表
        """
        return self.ranges


# 主程式
if __name__ == "__main__":
    print("設定初始範圍 (格式: [[H1, S1, V1], [H2, S2, V2]])")

    # 讓使用者輸入初始範圍
    initial_ranges = []
    while True:
        try:
            range_input = input("請輸入一組範圍 (或輸入 'done' 完成設定): ")
            if range_input.lower() == 'done':
                break

            # 將輸入轉換為範圍
            parsed_range = eval(range_input)  # 假設輸入格式為 [[H1, S1, V1], [H2, S2, V2]]
            if isinstance(parsed_range, list) and len(parsed_range) == 2:
                lower_bound, upper_bound = parsed_range
                if all(isinstance(v, int) for v in lower_bound + upper_bound) and len(lower_bound) == 3 and len(upper_bound) == 3:
                    initial_ranges.append([lower_bound, upper_bound])
                else:
                    raise ValueError
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print("請輸入有效範圍，例如: [[0, 88, 88], [18, 222, 173]]")

    if not initial_ranges:
        print("未設定範圍，程式結束。")
        exit()

    # 初始化動態範圍
    dynamic_range = DynamicHSVRange(initial_ranges)
    print("初始範圍:", dynamic_range.get_current_ranges())

    while True:
        # 接收使用者輸入的HSV值
        try:
            hsv_input = input("請輸入HSV值 (格式: [H, S, V]，或輸入 'exit' 結束): ")
            if hsv_input.lower() == 'exit':
                break

            # 將輸入轉換為HSV值
            hsv_value = eval(hsv_input)  # 假設輸入格式為 [H, S, V]
            if isinstance(hsv_value, list) and len(hsv_value) == 3 and all(isinstance(v, int) for v in hsv_value):
                # 檢查是否在範圍內，並調整範圍
                if dynamic_range.is_in_any_range(hsv_value):
                    print(f"HSV值 {hsv_value} 已在範圍內。")
                else:
                    dynamic_range.adjust_range(hsv_value)
                    print(f"HSV值 {hsv_value} 不在範圍內，已調整範圍。")

                # 顯示當前範圍
                print("當前範圍:", dynamic_range.get_current_ranges())
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print("請輸入有效的HSV值，例如: [10, 150, 200]")
