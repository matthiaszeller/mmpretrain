# dataset settings
dataset_type = 'IVOCTDataset'
data_root = 'data/shockwave/images-3D-cartesian'

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    # mean=[123.675, 116.28, 103.53],
    mean=[24.87],
    # std=[58.395, 57.12, 57.375],
    std=[53.36],
    #to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromZipFile', imdecode_backend='pillow', color_type='unchanged', stack_neighbour_slices=1),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=1., degree=360, seg_pad_val=0),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromZipFile', imdecode_backend='pillow', color_type='unchanged', stack_neighbour_slices=1),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #split='splits/train.txt',
        pipeline=train_pipeline))
