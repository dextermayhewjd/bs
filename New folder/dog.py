# _base_ = [
#     '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
#     '../../_base_/default_runtime.py'
# ]

# # model settings
# model = dict(
#     cls_head=dict(
#         type='TSNHead',
#         num_classes=101  # change from 400 to 101
#         ))
_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=20, # 修改为20帧 而不是8帧
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',#'joint_space_time'
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=18,#400 -> 18
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[127.5, 127.5, 127.5], # 像素值从 [0, 255] 的范围重新缩放到 [-1, 1]
        std=[127.5, 127.5, 127.5],
        format_shape='NCTHW'))#(ok and done)

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/dog/videos_train'
data_root_val = 'data/dog/videos_val'
ann_file_train = 'data/dog/dog_train_list_videos.txt'
ann_file_val = 'data/dog/dog_val_list_videos.txt'
ann_file_test = 'data/dog/dog_val_list_videos.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args), # whether or not( ok )
    # make sure the toal frame is 80 since it is 20 frame/sec * 4 
    dict(type='SampleFrames', clip_len=40, frame_interval=2, num_clips=1),# ok 
    dict(type='DecordDecode'),#ok
    dict(type='Resize', scale=(224, 224)),  # 直接调整为模型输入大小，不保持宽高比(ok)
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
    dict(type='Resize', scale=(224, 224)),  # 直接调整为模型输入大小，不保持宽高比(ok)
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
    dict(type='Resize', scale=(224, 224)),  # 直接调整为模型输入大小，不保持宽高比(ok)
    # works because flip does not matter 
    dict(type='Flip', flip_ratio=0.5),# ok (ok)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

NUM_CLASS=20

train_dataloader = dict(  # Config of train dataloader
    batch_size=32,  # Batch size of each single GPU during training
    num_workers=8,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
    sampler=dict(
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
        shuffle=True),  # Randomly shuffle the training data in each epoch
    dataset=dict(  # Config of train dataset
        type=dataset_type,
        multi_class=True,
        num_classes=NUM_CLASS,  # 更新为实际类别总数
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
        multi_class=True,
        num_classes=NUM_CLASS,  # 更新为实际类别总数
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
        multi_class=True,
        num_classes=NUM_CLASS,  # 更新为实际类别总数
        data_prefix=dict(img=data_root_val),  # Prefix of frame path
        pipeline=test_pipeline,
        test_mode=True))


val_evaluator = dict(type='MultiLabelClassificationMetrics')
test_evaluator = val_evaluator


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.005,
        momentum=0.9,
        weight_decay=1e-4, nesterov=True),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(interval=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)