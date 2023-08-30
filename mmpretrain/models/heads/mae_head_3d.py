
import torch
from mmpretrain.registry import MODELS
from .mae_head import MAEPretrainHead


@MODELS.register_module()
class MAEPretrainHead3D(MAEPretrainHead):
    """Head for MAE Pre-training.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
        in_chans (int): Image input channels. Defaults to 3.
        n_slice
    """

    def __init__(self,
                 n_slice: int = 3,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_slice = n_slice

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, n_slice * in_chans, H, W)`.

        Returns:
            torch.Tensor: Patchified images. The shape is
            :math:`(B, n_slice * L, \text{patch_size}^2 \times 3)`.
        """
        B, C, H, W = imgs.shape
        assert C == self.n_slice * self.in_chans
        # pull slices to batch dim
        x = imgs.reshape(B * self.n_slice, self.in_chans, H, W)
        # 2D patchify
        x = super().patchify(x)
        # push slices to patch dim
        return x.reshape(B, self.n_slice * x.shape[1], *x.shape[2:])

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_size}^2 \times 3)`.

        Returns:
            torch.Tensor: The shape is :math:`(B, 3, H, W)`.
        """
        B, L, D = x.shape
        # pull slices to batch dim
        x = x.reshape(B * self.n_slice, L // self.n_slice, D)
        # 2D unpatchify
        x = super().unpatchify(x)
        # push slices to channel dim
        return x.reshape(B, self.n_slice * x.shape[1], *x.shape[2:])
