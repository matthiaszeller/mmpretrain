# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseSelfSupervisor
from .beit import VQKD, BEiT, BEiTPretrainViT
from .byol import BYOL
from .cae import CAE, CAEPretrainViT, DALLEEncoder
from .densecl import DenseCL
from .eva import EVA
from .itpn import iTPN, iTPNHiViT
from .itpn_3d import iTPNHiViT3D
from .mae import MAE, MAEHiViT, MAEViT
from .maskfeat import HOGGenerator, MaskFeat, MaskFeatViT
from .milan import MILAN, CLIPGenerator, MILANViT
from .mixmim import MixMIM, MixMIMPretrainTransformer
from .moco import MoCo
from .mocov3 import MoCoV3, MoCoV3ViT
from .simclr import SimCLR
from .simmim import SimMIM, SimMIMSwinTransformer
from .simsiam import SimSiam
from .spark import SparK
from .swav import SwAV

__all__ = [
    'BaseSelfSupervisor',
    'BEiTPretrainViT',
    'VQKD',
    'CAEPretrainViT',
    'DALLEEncoder',
    'MAEViT',
    'MAEHiViT',
    'iTPNHiViT',
    'iTPNHiViT3D',
    'iTPN',
    'HOGGenerator',
    'MaskFeatViT',
    'CLIPGenerator',
    'MILANViT',
    'MixMIMPretrainTransformer',
    'MoCoV3ViT',
    'SimMIMSwinTransformer',
    'MoCo',
    'MoCoV3',
    'BYOL',
    'SimCLR',
    'SimSiam',
    'BEiT',
    'CAE',
    'MAE',
    'MaskFeat',
    'MILAN',
    'MixMIM',
    'SimMIM',
    'EVA',
    'DenseCL',
    'BarlowTwins',
    'SwAV',
    'SparK',
]
