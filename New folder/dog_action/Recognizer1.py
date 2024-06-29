# model_cfg = dict(
#     type='RecognizerZelda',
#     backbone=dict(type='BackBoneZelda'),
#     cls_head=dict(
#         type='ClsHeadZelda',
#         num_classes=2,
#         in_channels=128,
#         average_clips='prob'),
#     data_preprocessor = dict(
#         type='DataPreprocessorZelda',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375]))
# model = MODELS.build(model_cfg)

# # Train
# model.train()
# model.init_weights()
# data_batch_train = copy.deepcopy(batched_packed_results)
# data = model.data_preprocessor(data_batch_train, training=True)
# loss = model(**data, mode='loss')
# print('loss dict: ', loss)

# # Test
# with torch.no_grad():
#     model.eval()
#     data_batch_test = copy.deepcopy(batched_packed_results)
#     data = model.data_preprocessor(data_batch_test, training=False)
#     predictions = model(**data, mode='predict')
# print('Label of Sample[0]', predictions[0].gt_label)
# print('Scores of Sample[0]', predictions[0].pred_score)
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
        attention_type='space_only',
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
        format_shape='NCTHW'))