



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