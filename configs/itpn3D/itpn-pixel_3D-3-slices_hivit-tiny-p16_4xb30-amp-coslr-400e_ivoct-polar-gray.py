_base_ = [
    '../_base_/models/itpn_hivit-small-p16.py',
    '../_base_/datasets/ivoct-polar-gray_3D-3-slices_bs128_mae.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(
    batch_size=30,
    dataset=dict(skip_edge_slices=1)
)


model = dict(
    backbone=dict(
        type='iTPNHiViT3D',
        arch='tiny',
        n_slice=3,
        img_size=512,
        in_chans=1,
    ),
    neck=dict(
        type='iTPNPretrainDecoder3D',
        in_chans=1,
        patch_resolution=(32, 32),
        n_slice=3,
    ),
    head=dict(
        type='MAEPretrainHead3D',
        n_slice=3,
        in_chans=1,
    )
)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 120 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True
find_unused_parameters = False

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=120)
