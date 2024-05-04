from torch import nn
from model.embedding import PatchEmbedding
from model.transformer import TransformerEncoder
from model.network import ClassificationHead, ClassificationHead2

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 2,
                 emb_size: int = 192,
                 img_size: int = 32,
                 depth: int = 12,
                 numheads: int = 6,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 n_classes: int = 10,
                 if_cls: bool = True,
                 if_dis: bool = False):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size, if_cls, if_dis),
            TransformerEncoder(depth=depth, emb_size=emb_size, num_heads=numheads, drop_p=drop_p, forward_expansion=forward_expansion, forward_drop_p=forward_drop_p),
            ClassificationHead(emb_size, n_classes)
        )
