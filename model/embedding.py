import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import repeat


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 2, emb_size: int = 192, img_size: int = 32, if_cls: bool = True, if_dis: bool = False):
        super().__init__()
        self.if_cls = if_cls
        self.if_dis = if_dis
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.dis_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + int(if_cls) + int(if_dis), emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        dis_tokens = repeat(self.dis_token, '() n e -> b n e', b=b)
        # prepend the cls and dist token to the input
        if self.if_cls:
            x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        if self.if_dis:
            x = torch.cat([x, dis_tokens], dim=1)
        x += self.positions
        return x

