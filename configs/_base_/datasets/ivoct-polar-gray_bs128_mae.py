# dataset settings
dataset_type = 'IVOCTDataset'
data_root = 'data/shockwave/images-3D-polar'

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    # gray channel will be duplicated
    mean=[24.868, 24.868, 24.868],
    std=[53.360, 53.360, 53.360],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromZipFile', imdecode_backend='pillow', color_type='unchanged'),
    dict(type='DuplicateImageChannels', num_repeat=3),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRoll', axis=0),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #split='splits/train.txt',
        pipeline=train_pipeline))
