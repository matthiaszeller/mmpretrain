_base_ = [
    '../_base_/models/itpn_hivit-small-p16.py',
    '../_base_/datasets/ivoct-polar-gray_bs128_mae.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(
    batch_size=100,
)

model = dict(
    backbone=dict(
        arch='tiny',
        img_size=512,
        in_chans=1,
    ),
    neck=dict(
        in_chans=1,
        num_patches=1024, # (512 / 16) ** 2 = 1024
    ),
    head=dict(
        in_chans=1,
    )
)

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 400 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.01),
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

load_from='work_dirs/itpn-pixel_hivit-tiny-p16_4xb100-amp-coslr-400e_ivoct-polar-gray/epoch_400.pth'
resume = False
find_unused_parameters = False

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=400)

