import os

def rename_images_in_folder(folder_path, name, start_num):
    # 列出資料夾中的所有檔案
    files = os.listdir(folder_path)
    
    # 過濾圖片檔案，這裡只處理 jpg 和 png 檔案
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 逐一重命名
    for idx, file_name in enumerate(image_files):
        # 構建新的檔案名
        new_name = f"{name}{start_num + idx}{os.path.splitext(file_name)[1]}"
        
        # 取得完整路徑
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 重新命名檔案
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# 測試
folder_path = r"D:\Git\robotlearning\yoloface\target\result\data2"  # 請修改為實際的資料夾路徑
name = "Pcb"
start_num = 100

rename_images_in_folder(folder_path, name, start_num)
