# model = dict(  # 模型的配置
#     type='Recognizer2D',  # 识别器的类名
#     backbone=dict(  # 骨干网络的配置
#         type='ResNet',  # 骨干网络的名称
#         pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',# 预训练模型的 URL/网站
#         depth=50,  # ResNet 模型的深度
#         norm_eval=False),  # 是否在训练时将 BN 层设置为评估模式
#     cls_head=dict(  # 分类头的配置
#         type='TSNHead',  # 分类头的名称
#         num_classes=20,  # 要分类的类别数量。
#         in_channels=2048,  # 分类头的输入通道数。 ！！！！！
#         #存疑 in_channels (int): Number of channels in input feature.
#         spatial_type='avg',  # 空间维度池化的类型
#         consensus=dict(type='AvgConsensus', dim=1),  # 一致性模块的配置
#         dropout_ratio=0.4,  # dropout 层中的概率
#         init_std=0.01, # 线性层初始化的标准差值
#         average_clips='prob'),  # 平均多个剪辑结果的方法
#     data_preprocessor=dict(  # 数据预处理器的配置
#         type='ActionDataPreprocessor',  # 数据预处理器的名称
#         mean=[123.675, 116.28, 103.53],  # 不同通道的均值用于归一化
#         std=[58.395, 57.12, 57.375],  # 不同通道的标准差用于归一化
#         format_shape='NCHW'),  # 最终图像形状的格式
#     # 模型训练和测试设置
#     train_cfg=None,  # TSN 的训练超参数的配置
#     test_cfg=None)  # TSN 的测试超参数的配置

# dataset settings
dataset_type = 'VideoDataset'

data_root = 'data/dog/videos_train'
data_root_val = 'data/dog/videos_val'
data_root_test = 'data/dog/videos_test'

ann_file_train = 'data/dog/dog_train_list_videos.txt'
ann_file_val = 'data/dog/dog_val_list_videos.txt'
ann_file_test = 'data/dog/dog_test_list_videos.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args), # whether or not( ok )
    # make sure the toal frame is 80 since it is 20 frame/sec * 4 
    dict(type='SampleFrames', clip_len=40, frame_interval=2, num_clips=1),# ok 
    dict(type='DecordDecode'),#ok
    dict(type='Resize', scale=(224, 224),keep_ratio=False),  # 直接调整为模型输入大小，不保持宽高比(ok)
    # works because flip does not matter 
    dict(type='Flip', flip_ratio=0.5),# ok (ok)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args), # whether or not( ok )
    # make sure the toal frame is 80 since it is 20 frame/sec * 4 
    dict(type='SampleFrames', clip_len=40, frame_interval=2, num_clips=1,test_mode=True),# ok 
    dict(type='DecordDecode'),#ok
    dict(type='Resize', scale=(224, 224),keep_ratio=False),  # 直接调整为模型输入大小，不保持宽高比(ok)
    # works because flip does not matter 
    dict(type='Flip', flip_ratio=0.5),# ok (ok)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args), # whether or not( ok )
    # make sure the toal frame is 80 since it is 20 frame/sec * 4 
    dict(type='SampleFrames', clip_len=40, frame_interval=2, num_clips=1,test_mode=True),# ok 
    dict(type='DecordDecode'),#ok
    dict(type='Resize', scale=(224, 224),keep_ratio=False),  # 直接调整为模型输入大小，不保持宽高比(ok)
    # works because flip does not matter 
    dict(type='Flip', flip_ratio=0.5),# ok (ok)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


train_dataloader = dict(  # Config of train dataloader
    batch_size=32,  # Batch size of each single GPU during training
    num_workers=8,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
    sampler=dict(
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
        shuffle=True),  # Randomly shuffle the training data in each epoch
    dataset=dict(  # Config of train dataset
        type=dataset_type,
        ann_file=ann_file_train,  # Path of annotation file
        data_prefix=dict(img=data_root),  # Prefix of frame path
        pipeline=train_pipeline))       # ok
val_dataloader = dict(  # Config of validation dataloader
    batch_size=1,  # Batch size of each single GPU during validation
    num_workers=8,  # Workers to pre-fetch data for each single GPU during validation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of validation dataset
        type=dataset_type,
        ann_file=ann_file_val,  # Path of annotation file
        data_prefix=dict(img=data_root_val),  # Prefix of frame path
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(  # Config of test dataloader
    batch_size=32,  # Batch size of each single GPU during testing
    num_workers=8,  # Workers to pre-fetch data for each single GPU during testing
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of test dataset
        type=dataset_type,
        ann_file=ann_file_val,  # Path of annotation file
        data_prefix=dict(img=data_root_val),  # Prefix of frame path
        pipeline=test_pipeline,
        test_mode=True))