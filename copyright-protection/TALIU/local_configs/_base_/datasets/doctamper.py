# dataset settings
dataset_type = 'DocTamper'
data_root = "/dis/DS/ducanh/segment/DocTamperV1"
# data_root = "../datasets/DocTamperV1"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, tamaug=dict(enable=True, k=0.1, dynamic=dict(enable=False, iter=20000, total=160000))), # add tamaug if u set tamper_augment=True
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type='PhotoMetricDistortion'),
    dict(type='RGB2Gray'),
    
    #dict(type='RandomRotate', prob=1, degree=(10, 360))
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='ResizeToMultiple', size_divisor=32),
            # dict(type='RandomFlip', prob=1, direction='horizontal'),
            # dict(type='RandomFlip', prob=1, direction='vertical'),
            dict(type='RGB2Gray'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8, #batchsize
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotation/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test-fcd',
        ann_dir='annotation/test-fcd',
        # img_dir='images/validation',
        # ann_dir='annotation/validation',
        pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/test-fcd',
    #     ann_dir='annotation/test-fcd',
    #     pipeline=test_pipeline))
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test-scd',
        ann_dir='annotation/test-scd',
        pipeline=test_pipeline))
