from torch import nn
from einops.layers.torch import Reduce
import torch

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 192, n_classes: int = 10):
        super(ClassificationHead, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        return self.model(x[:, 0, :].squeeze())


class ClassificationHead2(nn.Module):
    def __init__(self, emb_size: int = 192, n_classes: int = 10, if_cls: bool = True, if_dis: bool = False):
        super(ClassificationHead2, self).__init__()
        self.layer = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.layer(x)
        cls_output = x[:, 0, :]
        cls_output = self.linear(cls_output)
        dis_output = x[:, -1, :]
        dis_output = self.linear(dis_output)

        result = torch.stack([cls_output, dis_output], dim=1)
        #input (b, n, e) output (b, 2, e)
        return result


