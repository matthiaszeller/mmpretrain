_base_ = [
    './mae_vit-base-p16_1xb100-amp-coslr-300e_ivoct-512x512.py',
]


model = dict(
    init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='work_dirs/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth')
)

# adjust the lr according to pretraining config
optim_wrapper = dict(
    optimizer=dict(
        lr=1.5e-4 * 100 / 256,
    )
)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=200)
