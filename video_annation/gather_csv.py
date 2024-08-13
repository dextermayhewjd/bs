import os
import shutil

def copy_csv_files(src_folder, dest_folder):
    # 确保目标文件夹存在，不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有子文件夹和文件
    for subdir, _, files in os.walk(src_folder):
        # 跳过目标文件夹
        if os.path.commonpath([subdir]) == os.path.commonpath([src_folder, dest_folder]):
            continue

        for file in files:
            # 找到所有的 csv 文件
            if file.lower().endswith('.csv'):
                src_file_path = os.path.join(subdir, file)
                dest_file_path = os.path.join(dest_folder, file)
                
                # 检查是否已经存在同名文件
                if os.path.exists(dest_file_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    new_dest_file_path = os.path.join(dest_folder, f"{base}_{counter}{extension}")
                    while os.path.exists(new_dest_file_path):
                        counter += 1
                        new_dest_file_path = os.path.join(dest_folder, f"{base}_{counter}{extension}")
                    dest_file_path = new_dest_file_path
                
                # 复制文件到目标文件夹
                shutil.copy2(src_file_path, dest_file_path)

    print(f"所有的 .csv 文件已复制到文件夹: {dest_folder}")

# 设置源文件夹和目标文件夹路径
source_folder = r'D:\dog_video\Feb\2_4_dixia\output_sensor'
destination_folder = r'D:\dog_video\all_sensor\sensor'

# 调用函数复制文件
copy_csv_files(source_folder, destination_folder)
