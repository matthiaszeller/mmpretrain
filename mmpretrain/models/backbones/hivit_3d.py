# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import build_norm_layer, to_2tuple
from .base_backbone import BaseBackbone
from .hivit import PatchEmbed, HiViT, BlockWithRPE, PatchMerge


class PatchEmbed3D(PatchEmbed):
    """PatchEmbed for HiViT.


    Args:
        n_slice (int): Number of slices stacked in the channel dimension
        img_size (int): Input image size.
        patch_size (int): Patch size. Defaults to 16.
        inner_patches (int): Inner patch. Defaults to 4.
        in_chans (int): Number of image input channels.
        embed_dim (int): Transformer embedding dimension.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        kernel_size (int): Kernel size.
        pad_size (int): Pad size.
    """

    def __init__(self, n_slice: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_slice = n_slice
        self.num_patches *= n_slice

    def forward(self, x):
        B_, C_, H, W = x.shape
        n_slice = C_ // (C := self.in_chans)
        assert n_slice * self.in_chans == C_, f'input channel {C_} cannot be divided by {self.in_chans}'

        # pull slices to batch dimension
        B = B_ * n_slice
        x = x.reshape(B_ * n_slice, C, H, W)

        # 2d patch embedding
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B,
            -1,
            patches_resolution[0],
            self.inner_patches,
            patches_resolution[1],
            self.inner_patches,
        # in the reshape, push slices to patch dimension
        ).permute(0, 2, 4, 3, 5, 1).reshape(B_, n_slice * num_patches, self.inner_patches,
                                            self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)

        return x


@MODELS.register_module()
class HiViT3D(HiViT):
    """HiViT.

    A PyTorch implement of: `HiViT: A Simple and More Efficient Design
    of Hierarchical Vision Transformer <https://arxiv.org/abs/2205.14949>`_.

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', and'base'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (int): The number of heads in attention
              modules of each stage.

        Defaults to 'tiny'.
        img_size (int): Input image size.
        patch_size (int): Patch size. Defaults to 16.
        inner_patches (int): Inner patch. Defaults to 4.
        in_chans (int): Number of image input channels.
        embed_dim (int): Transformer embedding dimension.
        depths (list[int]): Number of successive HiViT blocks.
        num_heads (int): Number of attention heads.
        stem_mlp_ratio (int): Ratio of MLP hidden dim to embedding dim
            in the first two stages.
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim in
            the last stage.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        ape (bool): If True, add absolute position embedding to
            the patch embedding.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
        patch_norm (bool): If True, use norm_cfg for normalization layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        kernel_size (int): Kernel size.
        pad_size (int): Pad size.
        layer_scale_init_value (float): Layer-scale init values. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
             Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 384,
                         'depths': [1, 1, 10],
                         'num_heads': 6}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'depths': [2, 2, 20],
                         'num_heads': 6}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 512,
                         'depths': [2, 2, 24],
                         'num_heads': 8}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 768,
                         'depths': [2, 2, 40],
                         'num_heads': 12}),
    }  # yapf: disable

    num_extra_tokens = 0

    def __init__(self,
                 arch='base',
                 n_slice: int = 3,
                 img_size=224,
                 patch_size=16,
                 inner_patches=4,
                 in_chans=3,
                 stem_mlp_ratio=3.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN'),
                 out_indices=[23],
                 ape=True,
                 ape_alpha=1.,
                 rpe=False,
                 patch_norm=True,
                 frozen_stages=-1,
                 kernel_size=None,
                 pad_size=None,
                 layer_scale_init_value=0.0,
                 init_cfg=None):
        super(HiViT3D, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']

        self.num_slice = n_slice
        self.num_stages = len(self.depths)
        self.ape = ape
        self.ape_alpha = ape_alpha
        self.rpe = rpe
        self.patch_size = patch_size
        self.num_features = self.embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_main_blocks = self.depths[-1]
        self.out_indices = out_indices
        self.out_indices[-1] = self.depths[-1] - 1

        img_size = to_2tuple(img_size) if not isinstance(img_size,
                                                         tuple) else img_size

        embed_dim = self.embed_dims // 2**(self.num_stages - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            n_slice=n_slice,
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_cfg=norm_cfg if patch_norm else None,
            kernel_size=kernel_size,
            pad_size=pad_size)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution

        if rpe:
            assert Hp == Wp, 'If you use relative position, make sure H == W '
            'of input size'

        # absolute position embedding
        if ape:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features))

        if rpe:
            raise NotImplementedError
            # get pair-wise relative position index for each token inside the
            # window
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index',
                                 relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(
            x.item()
            for x in torch.linspace(0, drop_path_rate,
                                    sum(self.depths) + sum(self.depths[:-1])))

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_i, stage_depth in enumerate(self.depths):
            is_main_stage = embed_dim == self.num_features
            nhead = self.num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage includes two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp,
                        embed_dim,
                        nhead,
                        ratio,
                        qkv_bias,
                        qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=next(dpr),
                        rpe=rpe,
                        norm_cfg=norm_cfg,
                        layer_scale_init_value=layer_scale_init_value,
                    ))
            if stage_i + 1 < self.num_stages:
                self.blocks.append(PatchMerge(embed_dim, norm_cfg))
                embed_dim *= 2

        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

        self.apply(self._init_weights)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed

        raise NotImplementedError
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(
            w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed
