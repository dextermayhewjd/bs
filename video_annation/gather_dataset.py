import os

# 定义主文件夹路径
main_folder = r'D:\dog_video\Feb\2_2_kt\checked'

summary_file = os.path.join(main_folder, 'dataset.txt')

# 打开汇总文件准备写入
with open(summary_file, 'w', encoding='utf-8') as outfile:
    # 遍历主文件夹中的所有子文件夹
    for subdir, dirs, files in os.walk(main_folder):
        for file in files:
            # 找到名为dataset.txt的文件
            if file == 'dataset.txt':
                file_path = os.path.join(subdir, file)
                # 读取每个dataset.txt的内容
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # 将内容写入汇总文件
                    outfile.write(infile.read())

print(f'汇总完成，汇总文件保存在：{summary_file}')