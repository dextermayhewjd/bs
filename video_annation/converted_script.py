import json
import argparse
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
import pandas as pd
      
def get_video_path(json_file_path):
    """
    根据json的路径以及名称 返回对应的视频路径和名称

    Args:
        json_file_path (_type_): D:\小狗视频\ 2_2_kt\ via \ 0205.json

    Returns:
        _type_: path : D:\小狗视频\ 2_2_kt\ video\ 020500.mp4
        
    eg.
        # example
        json_file_path = r'D:\小狗视频\2_1_kt\via_with_sensor\0200.json'
        video_path = get_video_path(json_file_path)
        print(video_path)
    """
    # 获取文件的基础路径和文件名
    base_path = os.path.dirname(json_file_path)  # D:\小狗视频\2_2_kt\via
    json_filename = os.path.basename(json_file_path)  # 0205.json
    
    # 修改基础路径中的 'via_with_sensor' 为 'video'
    video_base_path = base_path.replace('via_with_sensor', 'video')
    
    # 修改文件名: 删除扩展名，添加 '00', 更换扩展名为 '.mp4'
    video_filename = json_filename[:-5] + '00.mp4'  # 假设json文件名长度固定，或确保.json前面有足够字符
    
    # 合并新的路径
    video_path = os.path.join(video_base_path, video_filename)
    
    return video_path

def parse_actions(json_file_path):
    """
    从指定的JSON文件路径解析动作数据。
    返回一个列表，其中每个元素是一个包含开始时间、结束时间和标签的字典。
    Args: 
        json_file_path (_type_): D:\小狗视频\ 2_2_kt\ via
    Returns:
        a list of actions 
        eg.
        [
        {'start': 0, 'end': 5.06476, 'labels': ['stand']},
        {'start': 5.065, 'end': 13.06476, 'labels': ['walk']},
        {'start': 13.065, 'end': 14.37726, 'labels': ['stand']},
        {'start': 14.377, 'end': 15.39809, 'labels': ['walk']},
        {'start': 15.461, 'end': 18.73142, 'labels': ['stand']},
        {'start': 18.731, 'end': 19.58559, 'labels': ['walk']},
        {'start': 19.586, 'end': 20.29392, 'labels': ['stand']},
        {'start': 20.294, 'end': 20.62726, 'labels': ['walk']},
        {'start': 20.627, 'end': 22.64809, 'labels': ['stand']},
        {'start': 22.648, 'end': 24.85642, 'labels': ['walk']},
        {'start': 24.856, 'end': 26.27309, 'labels': ['stand']},
        {'start': 26.273, 'end': 28.89809, 'labels': ['walk']},
        {'start': 28.898, 'end': 44.50339, 'labels': ['stand']},
        {'start': 44.528, 'end': 52.28464, 'labels': ['mixed']},
        {'start': 52.285, 'end': 56.83785, 'labels': ['walk']},
        {'start': 56.838, 'end': 59.92118, 'labels': ['stand']},
        {'start': 13.005, 'end': 28.83785, 'labels': ['mixed']},
        {'start': 15.255, 'end': 59.92118, 'labels': ['sniff']}
        ]
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    actions = []
    for metadata_id, metadata_info in data["metadata"].items():
        action = {
            "start": metadata_info["z"][0],
            "end": metadata_info["z"][1],
            "labels": [metadata_info["av"]["1"]]  
        }
        actions.append(action)
    
    return actions
  
def sort_time_segments(time_segments):
  """
  Sort a list of time segments based on their 'start' time.

  Parameters:
  time_segments (list of dicts): A list where each dict represents a time segment
                                  with 'start', 'end', and 'labels' keys.

  Returns:
  list: A sorted list of the time segments based on their 'start' times.
  """
  # Sort the list of dictionaries by the 'start' key
  sorted_segments = sorted(time_segments, key=lambda x: x['start'])
  return sorted_segments

def handle_remaining_action(action, video_length, clip_length=4):
    """
    处理动作剩余部分，使其尽量以动作中心为中心，并考虑边界情况，生成一个clip长度的action
    """
    remaining_duration = action['end'] - action['start']
    center = (action['start'] + action['end']) / 2
    clip_start = max(0, center - clip_length / 2)
    clip_end = min(video_length, center + clip_length / 2)
    if clip_start == 0:
        clip_end = clip_length
    if clip_end == video_length:
        clip_start = video_length - clip_length
    clip_start = round(clip_start, 3)
    clip_end = round(clip_end, 3)
    return {'start': clip_start, 'end': clip_end, 'labels': action['labels']}

def get_video_stream_info(video_path, stream_index):
    # 使用 ffprobe 获取特定视频流的信息
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', f'v:{stream_index}',
        '-show_entries', 'stream=duration,nb_frames,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running ffprobe: {result.stderr}")
    
    # 解析 JSON 输出
    info = json.loads(result.stdout)
    stream_info = info['streams'][0]
    duration = float(stream_info['duration'])
    total_frames = int(stream_info['nb_frames'])
    # 获取帧率，并转换为浮点数
    r_frame_rate = stream_info['r_frame_rate']
    num, denom = map(int, r_frame_rate.split('/'))
    frame_rate = num / denom
    
    return duration, total_frames, frame_rate

def get_all_video_info(video_path):
    # 使用 ffprobe 获取视频信息，确定视频流数量
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=index',
        '-select_streams', 'v',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running ffprobe: {result.stderr}")
    
    # 解析 JSON 输出，获取所有视频流索引
    info = json.loads(result.stdout)
    stream_indices = [stream['index'] for stream in info['streams']]
    
    video_info = []
    for index in stream_indices:
        duration, total_frames, frame_rate = get_video_stream_info(video_path, index)
        video_info.append((duration, total_frames, frame_rate))
    
    return video_info


def list_all_streams(video_path):
    # 使用 ffprobe 获取所有流的信息
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=index,codec_type',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running ffprobe: {result.stderr}")
    
    # 解析 JSON 输出，获取所有流信息
    info = json.loads(result.stdout)
    return info['streams']

def detect_camera_type(video_path):
    # 列出所有流的信息
    streams = list_all_streams(video_path)
    video_stream_indices = [stream['index'] for stream in streams if stream['codec_type'] == 'video']
    audio_stream_indices = [stream['index'] for stream in streams if stream['codec_type'] == 'audio']
    
    # 判断摄像头类型
    if len(video_stream_indices) == 2 and video_stream_indices == [0, 1]:
        return "Type 1 Camera"
    elif len(video_stream_indices) == 2 and len(audio_stream_indices) == 1 and video_stream_indices == [0, 2]:
        return "Type 2 Camera"
    else:
        return "Unknown Camera Type"

def get_stream_0_duration(video_path):
    # 使用 ffprobe 获取索引为0的视频流的详细信息
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running ffprobe: {result.stderr}")
    
    # 解析 JSON 输出，获取视频流的持续时间
    info = json.loads(result.stdout)
    # print("Stream 0 detailed information:", json.dumps(info, indent=4))  # 添加调试信息，打印详细信息
    if 'streams' in info and len(info['streams']) > 0 and 'duration' in info['streams'][0]:
        return float(info['streams'][0]['duration'])
    return None



def adjust_time(time_segments, video_info):
    if len(video_info) < 2:
        print("视频流不足以比较，无法调整时间")
        return time_segments
    
    duration0 = video_info[0][0]  # Duration of video stream 1
    duration1 = video_info[1][0]  # Duration of video stream 2
    
    # Calculate duration difference
    duration_difference = duration0 - duration1
    
    adjusted_segments = []
    
    for segment in time_segments:
        start_time = segment['start'] - duration_difference
        end_time = segment['end'] - duration_difference
        
        # Ensure start time is not less than 0
        if start_time < 0:
            start_time = 0
        else:
            start_time = round(start_time, 5)  # Round to 5 decimal places
        
        # Ensure end time does not exceed duration of video stream 2
        if end_time > duration1:
            end_time = duration1
        else:
            end_time = round(end_time, 5)  # Round to 5 decimal places
        
        # Adjusted segment
        adjusted_segment = {
            'start': start_time,
            'end': end_time,
            'labels': segment['labels']
        }
        
        adjusted_segments.append(adjusted_segment)
    
    return adjusted_segments

def extract_clips(actions, video_length, clip_length=4):
    """
        输入一个action和该视频的video_length还有要分割的clip_length
        将超过clip_length的动作进行分割

        Args:
            json_file_path (_type_): D:\小狗视频\ 2_2_kt\ via

        Returns:
            _type_: path : D:\小狗视频\ 2_2_kt\ video\ 020500.mp4
    """
    clips = []
    
    for action in actions:
        action_duration = action['end'] - action['start']
        if action_duration > clip_length:
            # 对超过4秒的动作进行分割
            num_clips = int(action_duration // clip_length)
            for i in range(num_clips):
                clip_start = action['start'] + i * clip_length
                clip_end = clip_start + clip_length
                clip_start = round(clip_start, 3)
                clip_end = round(clip_end, 3)
                clips.append({'start': clip_start, 'end': clip_end, 'labels': action['labels'].copy()})
            # 处理可能的剩余部分
            remaining_start = action['start'] + num_clips * clip_length
            if action['end'] - remaining_start > 0:
#                 remaining_action = {'start': remaining_start, 'end': action['end'], 'labels': action['labels']}
#                 remaining_clip = handle_remaining_action(remaining_action, video_length, clip_length)
                remaining_clip = {'start': round(action['end']-clip_length,3),'end': round(action['end'],3), 'labels': action['labels']}
                clips.append(remaining_clip)
        else:
            # 直接处理整个动作
            clip = handle_remaining_action(action, video_length, clip_length)
            clips.append(clip)
    return clips

def update_clips_labels(clips, actions):
    new_clips = []  # 创建一个新的片段列表
    for clip in clips:
        new_labels = set(clip['labels'])  # 使用集合来避免重复的标签
        for action in actions:
            # 计算重叠的开始和结束时间
            overlap_start = max(clip['start'], action['start'])
            overlap_end = min(clip['end'], action['end'])
            # 确定重叠的持续时间
            overlap_duration = max(0, overlap_end - overlap_start)

            # 只有当实际存在重叠时才处理标签
            if overlap_duration > 0.3:
                # 直接将标签添加到集合中，重复的标签不会被添加
                new_labels.update(action['labels'])

        # 构建新的片段对象，包含更新后的标签列表
        new_clip = clip.copy()  # 复制原始片段对象
        new_clip['labels'] = list(new_labels)  # 更新标签列表
        new_clips.append(new_clip)  # 添加到新的片段列表中

    return new_clips
  
def validate_clips_with_actions(clips, actions):
    # 存放有错误标签的clips的索引和错误标签
    errors = []

    for clip_index, clip in enumerate(clips):
        for label in clip['labels']:
            # 检查该标签在原始actions中是否有时间重叠
            label_found = False
            for action in actions:
                if label in action['labels']:
                    # 计算重叠的开始和结束时间
                    overlap_start = max(clip['start'], action['start'])
                    overlap_end = min(clip['end'], action['end'])
                    # 确定重叠的持续时间
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > 0:
                        label_found = True
                        break

            if not label_found:
                errors.append((clip_index, label))

    return errors

def filter_clips_that_highly_alike(clips):
    filtered_clips = []
    
    # Loop through each clip in the clips list
    for clip in clips:
        # If the filtered_clips list is empty, add the first clip
        if not filtered_clips:
            filtered_clips.append(clip)
        else:
            # Check if the start time of the current clip is more than 0.5 seconds apart from the last added clip's start time
            if clip['start'] - filtered_clips[-1]['start'] > 0.8:
                filtered_clips.append(clip)

    # filtered_clips will now contain the clips filtered based on the specified criteria
    return filtered_clips
  
def get_video_length(video_path):
    """获取指定视频文件的长度（秒）。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # 将结果转换为浮点数秒
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video length: {e}")
        return None

def add_mixed_label(actions):
    for action in actions:
        labels = action['labels']
        if 'mixed' not in labels and 'walk' in labels and 'stand' in labels:
            labels.append('mixed')
            print(f"mixed was added in this snap: {action}")
    
    return actions

def processdata(distributed_dataset_file,centre_dataset_file,check_file,final_segments,video_file_path,output_dir,base_dir_name,json_file_name):
    action_names = [
    'play', 'lie_down', 'stand', 'walk', 'sit', 'pee_pose', 'pee', 
    'poop_pose', 'poop', 'sniff', 'vigorous', 'jump', 'mixed', 'iwp',
    'eat', 'lick_self','shake','drink'
                    ]

    # Open the dataset file in append mode
    with open(distributed_dataset_file, 'a') as distributed_f, open(centre_dataset_file, 'a') as centre_f,open(check_file, 'a') as check_f:
        # Process each segment
        for segment in final_segments:
            start_time = segment['start']
            labels = ",".join(str(i) for i, label in enumerate(action_names) if label in segment['labels'])
            segment_duration = 4  # Duration in seconds to extract 80 frames at 20 fps
            
            #20240202kt_020500_start0.0_2
            output_filename = f"{base_dir_name}_{json_file_name}_start{start_time}_{labels.replace(',', '_')}.mp4"
            
            output_path = os.path.join(output_dir,json_file_name,output_filename)
            
            # Write the directory name, filename and labels to the dataset file
            distributed_f.write(f"{output_filename} {labels.replace(',', ' ')}\n")
            centre_f.write(f"{output_filename} {labels.replace(',', ' ')}\n")
            
            # 获取动作名称
            action_labels = [label for label in segment['labels'] if label in action_names]
            action_names_str = ", ".join(action_labels)
            
            # 写入文件名和动作名称到检查文件
            check_f.write(f"{output_filename}: {action_names_str}\n")
            
            # FFmpeg command to extract 80 frames starting from the start time of the segment
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-i', video_file_path,
                '-vf', 'scale=1920:1080',
                '-an',  # Remove audio
                '-frames:v', '80',  # Extract exactly 80 frames
                '-fps_mode', 'vfr',  # Variable frame rate to handle frame extraction accurately
                output_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 检查命令是否成功执行
            if result.returncode != 0:
                print(f"Error processing {base_dir_name} {json_file_name}segment starting at {start_time}. Error: {result.stderr}")
            else:
                print(f"Processed segment {base_dir_name} {json_file_name} starting at {start_time} successfully.")


def extract_and_sort_sensor_times(actions):
    # 提取sensor和sensor_front的时间
    sensor_times = []
    for action in actions:
        if 'sensor' in action['labels']:
            sensor_times.append(action['start'])
            sensor_times.append(action['end'])
        if 'sensor_front' in action['labels']:
            sensor_times.append(action['start'])

    # 按时间从小到大排序
    sensor_times.sort()
    return sensor_times



def parse_sensor_time(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    epoch = datetime(2024, 1, 30)
    return (dt - epoch).total_seconds()

def parse_txt_file(file_path, via_file_name):
    '''
    give the file path of the sensor_ana and via file name 
    it will return the sensor_point_data which include 
    via_data = {
        'via': via_file_name,
        'csv_file': '',
        'rows': []
    }
    '''
    via_data = {
        'via': via_file_name,
        'csv_file': '',
        'rows': []
    }
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            via_file = parts[0]
            if via_file == via_file_name:
                if via_data['csv_file'] == '':
                    via_data['csv_file'] = parts[1]
                rows_data = sorted([int(parts[i]) for i in range(2, len(parts))])
                via_data['rows'].extend(rows_data)
                via_data['rows'] = sorted(via_data['rows'])
    
    return via_data  

def read_specific_row(file_path, row_number):
    # pandas的行索引是从0开始的，所以row_number需要减去1
    try:
        # 跳过row_number-1行，然后读取1行，header=None不使用标题行
        df = pd.read_csv(file_path, skiprows=row_number-1, nrows=1, header=None)
        # 将结果转换为字典形式返回
        return df.iloc[0].to_dict()
    except Exception as e:
        return f"Error: {e}"


def fromArray2Array(via_file_name, txt_file_path, csv_base_path):
    via_data = parse_txt_file(txt_file_path, via_file_name)
    # via_file_name = '0145'
    #txt_file_path = r'D:\dog_video\Feb\2_1_kt\sensor_ana.txt'
#      get data like this
#     {'via': '0145',
#      'csv_file': '2_1.csv',
#      'rows': [4734746, 4735223, 4737455]}
    rows_csv = via_data['rows']
    csv_file_path = f"{csv_base_path}\\{via_data['csv_file']}"
    # print(csv_file_path)
    sensor_times = []
    for row_number in rows_csv:
        row_data = read_specific_row(csv_file_path, row_number)
        if isinstance(row_data, dict):
            sensor_time_point = parse_sensor_time(row_data[0])
            sensor_times.append(sensor_time_point)
        else:
            print(row_data)
    # print(sensor_times)
    # [179101.043, 179105.789, 179127.989]
    return sensor_times,csv_file_path

def calculate_average_offset(sensor_times, video_times):
    # Ensure the input lists are numpy arrays
    sensor_times = np.array(sensor_times)
    video_times = np.array(video_times)

    # Calculate the differences between sensor times and video times
    time_diffs = sensor_times - video_times

    # Calculate the average time difference (offset)
    average_offset = np.mean(time_diffs)

    # Calculate the sensor time corresponding to video time 0
    sensor_time_at_video_zero = average_offset

    return average_offset, sensor_time_at_video_zero

def seconds_to_datetime(seconds):
    epoch = datetime(2024, 1, 30)
    return epoch + timedelta(seconds=seconds)

def add_seconds_to_timestamp(timestamp, seconds):
    # 将字符串形式的时间戳转换为datetime对象
    # timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    
    # 创建一个timedelta对象表示要添加的秒数
    time_delta = timedelta(seconds=seconds)
    
    # 将timedelta对象添加到datetime对象
    new_timestamp = timestamp + time_delta
    
    # 返回新的时间戳字符串，保持相同的格式
    return new_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

class TimeDataSearcher:
    def __init__(self):
        self.current_csv_file = None
        self.df = None

    def _read_csv(self, csv_file):
        self.df = pd.read_csv(csv_file, header=None)
        self.df['time'] = pd.to_datetime(self.df[0], format="%Y-%m-%d %H:%M:%S.%f")
        self.current_csv_file = csv_file

    def find_closest_time_row(self, csv_file, target_time_str):
        # 检查是否需要重新读取CSV文件
        if csv_file != self.current_csv_file:
            self._read_csv(csv_file)

        # 将目标时间转换为datetime对象
        target_time = datetime.strptime(target_time_str, "%Y-%m-%d %H:%M:%S.%f")

        # 计算每个时间点与目标时间点之间的绝对差值
        self.df['time_diff'] = abs(self.df['time'] - target_time)

        # 找到最小差值对应的行索引
        closest_row_index = self.df['time_diff'].idxmin()

        return closest_row_index

def read_and_save_partial_csv(csv_file, start_row,  output_txt_file,num_rows=400):
    # 使用skiprows参数跳过前面的行，使用nrows参数限制读取的行数
    df = pd.read_csv(csv_file, skiprows=range(start_row-1), nrows=num_rows, header=None)

    # 将DataFrame保存为TXT文件
    df.to_csv(output_txt_file, index=False, header=False, sep='\t')

def process_sensor_data(
                        distributed_sensor_dataset_file,
                        centre_sensor_dataset_file,
                        video_sensor_time,
                        via_file_name,
                        txt_file_path,
                        csv_base_path,
                        segments,
                        base_dir_name,
                        json_file_name,
                        output_dir,
                        check_sensor_file
                        ):
    '''
    Args:
        video_line_sensor_times 
    '''
    """
    根据json的路径以及名称 生成所有对应的传感器数据以及标注文件

    Args:
        video_line_sensor_times (array): [6.14527, 10.83277, 32.583]
        
        
    """
    action_names = [
    'play', 'lie_down', 'stand', 'walk', 'sit', 'pee_pose', 'pee', 
    'poop_pose', 'poop', 'sniff', 'vigorous', 'jump', 'mixed', 'iwp',
    'eat', 'lick_self','shake','drink'
                    ]
    
    
    # video_sensor_time 
    # eg [6.14527, 10.83277, 32.583]
    # first align the video line 0 first 
    # but where is the epoch 
    sensor_times_to_epoch , csv_file_path = fromArray2Array(via_file_name, txt_file_path, csv_base_path)
    #[179101.043, 179105.789, 179127.989]
    average_offset, sensor_time_at_video_zero = calculate_average_offset(sensor_times_to_epoch, video_sensor_time)
    # print(f"Average Offset: {average_offset}")
    # print(f"Sensor time at video time 0: {sensor_time_at_video_zero}")
    # here the actual_time_at_video_zero is datetime object 
    actual_time_at_video_zero = seconds_to_datetime(sensor_time_at_video_zero)
    # print(f"Actual time at video time 0: {actual_time_at_video_zero}")
    searcher = TimeDataSearcher()
    
    # 每个segment 的start time + actual_time_at_video_zero = target_time_str
    # 找到close row then 读取400行
    with open(distributed_sensor_dataset_file, 'a') as distributed_f, open(centre_sensor_dataset_file, 'a') as centre_f,open(check_sensor_file, 'a') as check_f:
        for segment in segments:
            start_time = segment['start']
            
            labels = ",".join(str(i) for i, label in enumerate(action_names) if label in segment['labels'])
            
            #20240202kt_020500_start0.0_2
            output_sensor_filename = f"{base_dir_name}_{json_file_name}_start{start_time}_{labels.replace(',', '_')}.csv"
            
            output_path = os.path.join(output_dir,json_file_name,output_sensor_filename)
            
            # Write the directory name, filename and labels to the dataset file
            distributed_f.write(f"{output_sensor_filename} {labels.replace(',', ' ')}\n")
            centre_f.write(f"{output_sensor_filename} {labels.replace(',', ' ')}\n")
            
            # 获取动作名称
            action_labels = [label for label in segment['labels'] if label in action_names]
            action_names_str = ", ".join(action_labels)
            
            # 写入文件名和动作名称到检查文件
            check_f.write(f"{output_sensor_filename}: {action_names_str}\n")
            
            
            
            
            real_life_start_time = add_seconds_to_timestamp(actual_time_at_video_zero, start_time)
            print(f"Actual time at video time {start_time}: {real_life_start_time}")
            print(real_life_start_time)
            closest_row = searcher.find_closest_time_row(csv_file_path, real_life_start_time)
            print(f"Closest row index for {real_life_start_time}: {closest_row}")
            read_and_save_partial_csv(csv_file_path,
                            closest_row, output_path)
            print(f"Data saved to {output_path}")

def remove_sensor_labels(actions):
    # 创建一个新的列表来存储不包含sensor和sensor_front的项
    filtered_actions = []
    for action in actions:
        # 如果action的labels中不包含sensor和sensor_front，则添加到filtered_actions中
        if 'sensor' not in action['labels'] and 'sensor_front' not in action['labels']:
            filtered_actions.append(action)
    return filtered_actions

def main():
    # check whether there are enough text to process the main function
    parser = argparse.ArgumentParser(description="处理文件的脚本。")

    # 必选参数：文件名
    parser.add_argument('js_file_path', help='文件名参数，指定要处理的文件 D:/dog_video/Feb/2_2_kt/via/0205.json must include the ''.json')
    # 可选参数：输出目录，默认值为 'output'
    parser.add_argument('--output_dir', default='output', help='输出目录，指定处理后文件的存储位置（默认：output）。')
    # 可选参数：数据集目录，默认值为 'dataset'
    parser.add_argument('--dataset_dir', default='dataset', help='数据集目录，指定数据集存储的位置（默认：dataset）。')
    
    args = parser.parse_args()
    '''
            Data Directory Structure
        
        dog_video/Feb/2_2_kt (base_dir)
        │
        ├── video(video_dir)
        │   ├── 020500.mp4  # Raw video files
        │   └── .....       # Additional video files
        │
        ├── via
        │   ├── 0205.json   # Annotations in JSON format
        │   ├── ....        # Json_file_path
        │   └── ....        # Additional annotation files
        │
        ├── output(output_dir)
        │   ├──0205
        │   │   ├── 20240202kt_020500_start0.0_2 
        │   │   ├── .......
        │   │   ├── # Processed output files
        │   │   ├── # checked file 
        │   │   └── (distributed)dataset.txt
        │   ├──.......      # Processed output files time
        │   ├──.......
        │   └──dataset.txt  # Summary of datasets or additional information
        │ 
        ├── csv file(csv file)
        │   ├──2-1.csv
        │   └──2-3.csv
        │ 
        └──sensor_ana(keep all the sensor timestamp's row in csv fil)
    '''
    json_file_path = args.js_file_path
    # 获取文件名
    json_time_file_name = os.path.basename(json_file_path)
    
    # 去掉文件扩展名
    json_time_file_name_without_extension = os.path.splitext(json_time_file_name)[0]
    # print("json_time_file_name_without_extension:",
    # json_time_file_name_without_extension)# 基于 JSON 文件路径的目录
    # print(json_time_file_name_without_extension) # should be 0205 
    base_dir = os.path.dirname(os.path.dirname(json_file_path))  # 获取上级目录的上级目录  
    base_dir_name = os.path.basename(base_dir)
    sensor_file = os.path.join(base_dir,'sensor_ana.txt')
    # print("base_dir_name:",       base_dir_name)
    
    
    # 自动生成其他路径和名称
    video_dir = os.path.join(base_dir, 'video')
    via_dir = os.path.join(base_dir, 'via')
    output_dir = os.path.join(base_dir, 'output')
    output_sensor_dir = os.path.join(base_dir, 'output_sensor')    
    csv_dir = os.path.join(base_dir,'csv')
    # for video 
    centre_dataset_file = os.path.join(output_dir, 'dataset.txt')
    distributed_dataset_file = os.path.join(output_dir, json_time_file_name_without_extension,'dataset.txt')
    distributed_output_dir = os.path.join(output_dir, json_time_file_name_without_extension)
    # 创建新的文本文件用于记录文件名和动作名称
    check_file  = os.path.join(distributed_output_dir,"checklist.txt")
    
    # for sensor data
    centre_sensor_dataset_file = os.path.join(output_sensor_dir, 'dataset.txt')
    distributed_sensor_dataset_file = os.path.join(output_sensor_dir, json_time_file_name_without_extension,'dataset.txt')
    distributed_output_sensor_dir = os.path.join(output_sensor_dir, json_time_file_name_without_extension)
    check_sensor_file  = os.path.join(distributed_output_sensor_dir,"checklist.txt")



    # 确保文件夹存在，如果不存在则创建它
    if not os.path.exists(distributed_output_sensor_dir):
        os.makedirs(distributed_output_sensor_dir)
        print(f"{distributed_output_sensor_dir}                   不存在，已创建")
    else:
        print(f"{distributed_output_sensor_dir}                   存在，不创建")
    # 检查和创建centre_sensor_dataset_file
    if not os.path.exists(centre_sensor_dataset_file):
        os.makedirs(os.path.dirname(centre_sensor_dataset_file), exist_ok=True)
        with open(centre_sensor_dataset_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{centre_sensor_dataset_file}            不存在，已创建")
    else:
        print(f"{centre_sensor_dataset_file}            存在，不创建")
    # 检查和创建distributed_dataset_file
    if not os.path.exists(distributed_sensor_dataset_file):
        os.makedirs(os.path.dirname(distributed_sensor_dataset_file), exist_ok=True)
        with open(distributed_sensor_dataset_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{distributed_sensor_dataset_file}       不存在，已创建")
    else:
        print(f"{distributed_sensor_dataset_file}       存在，不创建")
        
    if not os.path.exists(check_sensor_file):
        os.makedirs(os.path.dirname(check_sensor_file), exist_ok=True)
        with open(check_sensor_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{check_sensor_file}       不存在，已创建")
    else:
        print(f"{check_sensor_file}       存在，不创建")
    
    
    # 确保文件夹存在，如果不存在则创建它
    if not os.path.exists(distributed_output_dir):
        os.makedirs(distributed_output_dir)
        print(f"{distributed_output_dir}                   不存在，已创建")
    else:
        print(f"{distributed_output_dir}                   存在，不创建")
    # 检查和创建centre_dataset_file
    if not os.path.exists(centre_dataset_file):
        os.makedirs(os.path.dirname(centre_dataset_file), exist_ok=True)
        with open(centre_dataset_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{centre_dataset_file}            不存在，已创建")
    else:
        print(f"{centre_dataset_file}            存在，不创建")
    # 检查和创建distributed_dataset_file
    if not os.path.exists(distributed_dataset_file):
        os.makedirs(os.path.dirname(distributed_dataset_file), exist_ok=True)
        with open(distributed_dataset_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{distributed_dataset_file}       不存在，已创建")
    else:
        print(f"{distributed_dataset_file}       存在，不创建")
        
    if not os.path.exists(check_file):
        os.makedirs(os.path.dirname(check_file), exist_ok=True)
        with open(check_file, 'w') as file:
            file.write("")  # 创建一个空文件
        print(f"{check_file}       不存在，已创建")
    else:
        print(f"{check_file}       存在，不创建")
    
    
    
    
    #change D:\小狗视频\2_2_kt\via to D:\小狗视频\2_2_kt\video\020500.mp4
    video_file_path = get_video_path(json_file_path)
    
    # 打印结果以验证
    
    # print("Base                     directory path :  ", base_dir)
    # print("Video                    directory path :  ", video_dir)
    # print("Via                      directory path :  ", via_dir)
    # print("Output                    directory path:  ", output_dir)
    # print("Output Sensor             directory path:  ", output_sensor_dir)
    # print("Distributed               directory path:  ", distributed_output_dir)
    # print("csv                       directory path:  ", csv_dir)
    # print("JSON                           file path:  ", json_file_path)
    # print("Centre Dataset                 file path:  ", centre_dataset_file)
    # print("Distributed Dataset            file path:  ", distributed_dataset_file)
    # print("check                          file path:  ", check_file)   
    # print("Centre Sensor Datase           file path:  ", centre_sensor_dataset_file)
    # print("Distributed  Sensor Dataset    file path:  ", distributed_sensor_dataset_file)
    # print("sensor_check                   file path:  ", check_sensor_file) 
    # print("sensor ana                     file path:  ", sensor_file) 

    
    # print(json_file_path)
    # print(video_file_path)
    
    
    # detect camera type
    camera_type = detect_camera_type(video_file_path)
    if camera_type == "Type 1 Camera":
        # 执行与 Type 1 Camera 相关的操作
        print("Executing actions for Type 1 Camera")
        video_info = get_all_video_info(video_file_path)
        
        actions = parse_actions(json_file_path)
        
        
        #edit time so the start time and end time both subtract the time difference
        adjusted_time_segments0 = adjust_time(actions,video_info)
        
        video_line_sensor_times = extract_and_sort_sensor_times(adjusted_time_segments0)
        # print(video_line_sensor_times)
        # eg [6.14527, 10.83277, 32.583]
        adjusted_time_segments1 = remove_sensor_labels(adjusted_time_segments0)
        # for no_sensor_action in adjusted_time_segments1:
        #     print(no_sensor_action)
        # sort the time
        sorted_actions = sort_time_segments(adjusted_time_segments1)
        # for sorted_action in sorted_actions: 
        #     print(sorted_action)
        video_length = video_info[1][0] # Duration of video stream 2
        extracted_clips = extract_clips(sorted_actions, video_length,4)
        
        updated_clips_with_label = update_clips_labels(extracted_clips,sorted_actions)
        sorted_updated_clips_with_label = sort_time_segments(updated_clips_with_label)
        
        filtered_clips_without_adding_mixed = filter_clips_that_highly_alike(sorted_updated_clips_with_label)
        # if both walk and stand in the snap
        filtered_clips = add_mixed_label(filtered_clips_without_adding_mixed)
        
        # Assume the frame rate is 30 frames per second
        frame_rate = 20
        frame_duration = 1 / frame_rate
        
        segments = filtered_clips
        adjusted_segments = []
        for segment in segments:
            adjusted_start = (segment['start'] // frame_duration) * frame_duration
            adjusted_start = round(adjusted_start, 2)
            adjusted_segments.append({**segment, 'start': adjusted_start})
        # pass the test 
        # for segment in adjusted_segments:
        #     print(segment)
            # 确保输出目录存在
        print(video_line_sensor_times)
    
    

    elif camera_type == "Type 2 Camera":
        # 执行与 Type 2 Camera 相关的操作
        print("Executing actions for Type 2 Camera (...)")
        actions = parse_actions(json_file_path)
        
        time_segments0 = actions
        
        video_line_sensor_times = extract_and_sort_sensor_times(time_segments0)
        print(video_line_sensor_times)
        
        # eg [6.14527, 10.83277, 32.583]
    #    #edit time so the start time and end time both subtract the time difference
    #     adjusted_time_segments0 = adjust_time(actions,video_info)
        
    #     video_line_sensor_times = extract_and_sort_sensor_times(adjusted_time_segments0)
    #     # print(video_line_sensor_times)
    #     # eg [6.14527, 10.83277, 32.583]
    
        time_segments1 = remove_sensor_labels(time_segments0)
        for no_sensor_action in time_segments1:
            print(no_sensor_action)
        # sort the time
        sorted_actions = sort_time_segments(time_segments1)
        for sorted_action in sorted_actions: 
            print(sorted_action)
        
        duration = get_stream_0_duration(video_file_path)
        
        video_length = duration
        print(video_length)
        extracted_clips = extract_clips(sorted_actions, video_length,4)
        
        updated_clips_with_label = update_clips_labels(extracted_clips,sorted_actions)
        sorted_updated_clips_with_label = sort_time_segments(updated_clips_with_label)
        
        filtered_clips_without_adding_mixed = filter_clips_that_highly_alike(sorted_updated_clips_with_label)
        # # if both walk and stand in the snap
        filtered_clips = add_mixed_label(filtered_clips_without_adding_mixed)
        
        # # Assume the frame rate is 30 frames per second
        frame_rate = 20
        frame_duration = 1 / frame_rate
        
        segments = filtered_clips
        adjusted_segments = []
        for segment in segments:
            adjusted_start = (segment['start'] // frame_duration) * frame_duration
            adjusted_start = round(adjusted_start, 2)
            adjusted_segments.append({**segment, 'start': adjusted_start})
        # # pass the test 
        # # for segment in adjusted_segments:
        # #     print(segment)
        #     # 确保输出目录存在
        # print(video_line_sensor_times)
    
    else:
        print("Unknown Camera Type. No actions to perform.")
#     #[0][duration, total_frames, frame_rate]
    # video_info = get_all_video_info(video_file_path)
    
    # actions = parse_actions(json_file_path)
    
    
    # #edit time so the start time and end time both subtract the time difference
    # adjusted_time_segments0 = adjust_time(actions,video_info)
    
    # video_line_sensor_times = extract_and_sort_sensor_times(adjusted_time_segments0)
    # # print(video_line_sensor_times)
    # # eg [6.14527, 10.83277, 32.583]
    # adjusted_time_segments1 = remove_sensor_labels(adjusted_time_segments0)
    # # for no_sensor_action in adjusted_time_segments1:
    # #     print(no_sensor_action)
    # # sort the time
    # sorted_actions = sort_time_segments(adjusted_time_segments1)
    # # for sorted_action in sorted_actions: 
    # #     print(sorted_action)
    # video_length = video_info[1][0] # Duration of video stream 2
    # extracted_clips = extract_clips(sorted_actions, video_length,4)
    
    # updated_clips_with_label = update_clips_labels(extracted_clips,sorted_actions)
    # sorted_updated_clips_with_label = sort_time_segments(updated_clips_with_label)
    
    # filtered_clips_without_adding_mixed = filter_clips_that_highly_alike(sorted_updated_clips_with_label)
    # # if both walk and stand in the snap
    # filtered_clips = add_mixed_label(filtered_clips_without_adding_mixed)
    
    # # Assume the frame rate is 30 frames per second
    # frame_rate = 20
    # frame_duration = 1 / frame_rate
    
    # segments = filtered_clips
    # adjusted_segments = []
    # for segment in segments:
    #     adjusted_start = (segment['start'] // frame_duration) * frame_duration
    #     adjusted_start = round(adjusted_start, 2)
    #     adjusted_segments.append({**segment, 'start': adjusted_start})
    # # pass the test 
    # # for segment in adjusted_segments:
    # #     print(segment)
    #     # 确保输出目录存在
    # print(video_line_sensor_times)
    
    

    
    # process_sensor_data(distributed_sensor_dataset_file=distributed_sensor_dataset_file,
    #                     centre_sensor_dataset_file=centre_sensor_dataset_file,
    #                     video_sensor_time = video_line_sensor_times,
    #                     via_file_name= json_time_file_name_without_extension,
    #                     txt_file_path= sensor_file,
    #                     csv_base_path= csv_dir,
    #                     segments=adjusted_segments,
    #                     base_dir_name=base_dir_name,
    #                     json_file_name=json_time_file_name_without_extension,
    #                     output_dir=output_sensor_dir,
    #                     check_sensor_file=check_sensor_file
    #                     )
    
    processdata(distributed_dataset_file=distributed_dataset_file,
                centre_dataset_file=centre_dataset_file,
                final_segments=adjusted_segments,
                video_file_path= video_file_path,
                output_dir=output_dir,
                base_dir_name= base_dir_name,
                json_file_name= json_time_file_name_without_extension,
                check_file=check_file
                )



    

if __name__ == '__main__':
    main()