_base_ = [
    './mae_vit-base-p16_1xb128-amp-coslr-300e_ivoct-512x512.py',
]


model = dict(
    init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='work_dirs/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth')
)

# adjust the lr according to pretraining config
optim_wrapper = dict(
    optimizer=dict(
        lr=1.5e-4 * 128 / 4096,
    )
)
