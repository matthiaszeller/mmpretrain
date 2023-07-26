_base_ = [
    './mae_vit-base-p16_1xb100-amp-coslr-300e_ivoct-512x512.py',
]


model = dict(
    init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='work_dirs/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth')
)

# adjust the lr according to pretraining config
optim_wrapper = dict(
    optimizer=dict(
        lr=1.5e-4 * 100 / 256,
    )
)
