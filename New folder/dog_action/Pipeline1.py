# for dog action recognition
train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=40, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=40, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=40, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]



# standard
# train_pipeline = [
#     dict(type='DecordInit',),
#     dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]


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

