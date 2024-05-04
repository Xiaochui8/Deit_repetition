# Deit_repetition
repete deit model

## Background 

ViT

- [ ] 归纳偏置?
- [ ] 训练的时候用一个隐藏层的MLP，Finetune的时候用一个线性层

class token是embedded得到

1D position embedding 加入到patch embedding

layer norm在每一个block前



pretrain：

ImageNet    ImageNet-21k    JFT300M这三个数据

Adam：β1 = 0.9, β2 = 0.999, a batch size of 4096  weight decay of 0.1





fine tune：





ViT的Pytorch代码：

[FrancescoSaverioZuppichini/ViT: Implementing Vi(sion)T(transformer) (github.com)](https://github.com/FrancescoSaverioZuppichini/ViT)

[vit-pytorch/vit_pytorch/vit.py at main · lucidrains/vit-pytorch (github.com)](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

[【超详细】初学者包会的Vision Transformer（ViT）的PyTorch实现代码学习_vision transformer代码-CSDN博客](https://blog.csdn.net/CHENCHENCHEN0526/article/details/121311456)

[【小白学习笔记】Pytorch之Vision Transformer(ViT)，CIFAR10图像分类, Colab 源码分享 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/448687823)



蒸馏

- [x] [知识蒸馏技术（教师学生模型） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/442457531)



散度和交叉熵

- [x] [Kullback-Leibler(KL)散度介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/100676922)(这个讲得还挺有意思的，虽然Loss计算很简单)

- [x] [交叉熵和KL散度有什么区别？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/292434104)



#### ResNet

[ResNet 详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/550360817)



### 问题

- [ ] position embedding在Transformer之前添加就可以了吗？

- [ ] 为什么class token和distillation token要分开，不能都用class token计算loss吗？

- [ ] 硬蒸馏和软蒸馏的区别？（散度和交叉熵的区别？）

  

- [ ] 怎么初始化参数

  initialize the weights with a truncated normal distribution

- [x] 数据和模型怎么放到GPU上面跑

- [ ] class token和position从embedding怎么样才好

- [x] distillation是在pretrain和finetune之后吗？（distillation两种，pretrain和finetune）



### 数据

https://pytorch.org/vision/0.17/generated/torchvision.datasets.CIFAR10.html



### 实现细节

train过程，head用linear classifier



### 参数

τ = 3.0 and λ = 0.1

epochs = 300

optimizer为



### 代码模板

[深度学习pytorch训练代码模板(个人习惯)(一整套流程完整运行) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/596449652)

[pytorch入门与实践（2）：小试牛刀之CIFAR-10图像分类 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/344615710)

[PyTorch 13.模型保存与加载，checkpoint - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/148159709)



### 技巧

[einops.repeat, rearrange, reduce优雅地处理张量维度-CSDN博客](https://blog.csdn.net/qq_37297763/article/details/120348764)

mode ema：[【炼丹技巧】指数移动平均（EMA）的原理及PyTorch实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/68748778)

DDP启动：

节约显存：

* fp16：[如何使用 PyTorch 进行半精度、混(合)精度训练_pytorch半精度训练-CSDN博客](https://blog.csdn.net/qq_44089890/article/details/130471991)

* flash-attn：
* xformer：
* checkpointing：
  



防止过拟合：dropout weight_decay drop_path 数据增强





### 依赖

```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```









### 结果

一开始，是随机初始化的

![image-20240414203812111](D:\Code\SCH\Deit_repetition\assets\image-20240414203812111.png)





结果：

```python
resnet18
accuracy = 89%
epoch = 33

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)



100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 31.902%, loss = 1.7865706067865768
100%|██████████| 79/79 [00:03<00:00, 20.96it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 0, valid acc = 49.060%, loss = 1.400978040091599
100%|██████████| 782/782 [00:44<00:00, 17.43it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.708%, loss = 1.1118888021125208
100%|██████████| 79/79 [00:03<00:00, 21.61it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 1, valid acc = 60.320%, loss = 1.1393954934953134
100%|██████████| 782/782 [00:44<00:00, 17.40it/s]
train acc = 70.369%, loss = 0.8408251243936437
100%|██████████| 79/79 [00:03<00:00, 21.44it/s]
epoch = 2, valid acc = 73.730%, loss = 0.7649450373800495
100%|██████████| 782/782 [00:45<00:00, 17.37it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.948%, loss = 0.6952720750552004
100%|██████████| 79/79 [00:03<00:00, 20.68it/s]
epoch = 3, valid acc = 75.630%, loss = 0.7037808619722535
100%|██████████| 782/782 [00:45<00:00, 17.32it/s]
train acc = 79.014%, loss = 0.6029253234262661
100%|██████████| 79/79 [00:03<00:00, 21.53it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 4, valid acc = 75.790%, loss = 0.7060822897319552
100%|██████████| 782/782 [00:44<00:00, 17.38it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.268%, loss = 0.5429211868273328
100%|██████████| 79/79 [00:03<00:00, 21.61it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 5, valid acc = 78.910%, loss = 0.614199358073971
100%|██████████| 782/782 [00:44<00:00, 17.42it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.042%, loss = 0.4919624843865709
100%|██████████| 79/79 [00:03<00:00, 21.47it/s]
epoch = 6, valid acc = 79.920%, loss = 0.5811659186323986
100%|██████████| 782/782 [00:44<00:00, 17.39it/s]
train acc = 84.190%, loss = 0.4582053038203503
100%|██████████| 79/79 [00:03<00:00, 21.58it/s]
epoch = 7, valid acc = 79.190%, loss = 0.6134937039658993
100%|██████████| 782/782 [00:44<00:00, 17.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.457%, loss = 0.4242316346301142
100%|██████████| 79/79 [00:03<00:00, 21.61it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 8, valid acc = 81.470%, loss = 0.5396530447126944
100%|██████████| 782/782 [00:48<00:00, 16.10it/s]
train acc = 86.156%, loss = 0.40053164274872416
100%|██████████| 79/79 [00:03<00:00, 20.93it/s]
epoch = 9, valid acc = 81.400%, loss = 0.5619408752344832
100%|██████████| 782/782 [00:45<00:00, 17.19it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 89.166%, loss = 0.31287718865343983
100%|██████████| 79/79 [00:03<00:00, 21.09it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 10, valid acc = 85.460%, loss = 0.4325492165133923
100%|██████████| 782/782 [00:45<00:00, 17.17it/s]
train acc = 89.998%, loss = 0.28857769764712093
100%|██████████| 79/79 [00:03<00:00, 21.28it/s]
epoch = 11, valid acc = 85.090%, loss = 0.443957270512098
100%|██████████| 782/782 [00:45<00:00, 17.35it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.469%, loss = 0.27469937180352333
100%|██████████| 79/79 [00:03<00:00, 21.63it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 12, valid acc = 85.640%, loss = 0.42727993994574004
100%|██████████| 782/782 [00:44<00:00, 17.40it/s]
train acc = 90.777%, loss = 0.2654043020933028
100%|██████████| 79/79 [00:03<00:00, 21.54it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 13, valid acc = 85.260%, loss = 0.4267612133599535
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 91.200%, loss = 0.2552633989135475
100%|██████████| 79/79 [00:03<00:00, 19.96it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 14, valid acc = 85.450%, loss = 0.43095531550389304
100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
train acc = 91.388%, loss = 0.245995516848305
100%|██████████| 79/79 [00:03<00:00, 21.22it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 15, valid acc = 84.760%, loss = 0.45742145388186733
100%|██████████| 782/782 [00:45<00:00, 17.26it/s]
train acc = 91.836%, loss = 0.23387674853929777
100%|██████████| 79/79 [00:03<00:00, 21.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 16, valid acc = 85.400%, loss = 0.43207742528447624
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 92.002%, loss = 0.22675887289483224
100%|██████████| 79/79 [00:03<00:00, 21.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 17, valid acc = 86.320%, loss = 0.4224833847978447
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 92.170%, loss = 0.2219058041034452
100%|██████████| 79/79 [00:03<00:00, 21.22it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 18, valid acc = 86.180%, loss = 0.42064303140851517
100%|██████████| 782/782 [00:45<00:00, 17.29it/s]
train acc = 92.574%, loss = 0.2112596679736129
100%|██████████| 79/79 [00:03<00:00, 21.21it/s]
epoch = 19, valid acc = 84.280%, loss = 0.4855461573298973
100%|██████████| 782/782 [00:45<00:00, 17.34it/s]
train acc = 94.542%, loss = 0.1565014896604716
100%|██████████| 79/79 [00:03<00:00, 21.31it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 20, valid acc = 87.440%, loss = 0.3939099900330169
100%|██████████| 782/782 [00:44<00:00, 17.40it/s]
train acc = 95.164%, loss = 0.1386538569998863
100%|██████████| 79/79 [00:03<00:00, 21.37it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 21, valid acc = 87.660%, loss = 0.394845533974563
100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
train acc = 95.377%, loss = 0.13408671553387208
100%|██████████| 79/79 [00:03<00:00, 21.11it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 22, valid acc = 87.380%, loss = 0.41543070255201076
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
train acc = 95.573%, loss = 0.12800252745809304
100%|██████████| 79/79 [00:03<00:00, 21.10it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 23, valid acc = 87.580%, loss = 0.41237166580520096
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 95.665%, loss = 0.125700734774856
100%|██████████| 79/79 [00:03<00:00, 21.25it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 24, valid acc = 87.670%, loss = 0.42142693490921695
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 95.780%, loss = 0.1198792719398923
100%|██████████| 79/79 [00:03<00:00, 21.10it/s]
epoch = 25, valid acc = 87.420%, loss = 0.4160702800071692
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 95.896%, loss = 0.1157720364211008
100%|██████████| 79/79 [00:03<00:00, 21.38it/s]
epoch = 26, valid acc = 87.280%, loss = 0.43386475855036627
100%|██████████| 782/782 [00:45<00:00, 17.28it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 96.050%, loss = 0.11335518439312267
100%|██████████| 79/79 [00:03<00:00, 21.18it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 27, valid acc = 87.510%, loss = 0.4129493459116054
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
train acc = 96.058%, loss = 0.11261923465630054
100%|██████████| 79/79 [00:03<00:00, 21.23it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 28, valid acc = 87.370%, loss = 0.4378138475025756
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
train acc = 96.257%, loss = 0.1066974813780745
100%|██████████| 79/79 [00:03<00:00, 21.12it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 29, valid acc = 87.270%, loss = 0.42902768469309505
100%|██████████| 782/782 [00:45<00:00, 17.33it/s]
train acc = 97.364%, loss = 0.07646358121053108
100%|██████████| 79/79 [00:03<00:00, 21.29it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 30, valid acc = 88.020%, loss = 0.41969865485082697
100%|██████████| 782/782 [00:45<00:00, 17.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 97.798%, loss = 0.06564684528643099
100%|██████████| 79/79 [00:03<00:00, 21.20it/s]
epoch = 31, valid acc = 88.360%, loss = 0.4174492029072363
100%|██████████| 782/782 [00:45<00:00, 17.26it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 97.907%, loss = 0.06221010456399997
100%|██████████| 79/79 [00:03<00:00, 21.04it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 32, valid acc = 88.330%, loss = 0.43312546856041195
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 97.956%, loss = 0.0594968045037478
100%|██████████| 79/79 [00:03<00:00, 21.07it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 33, valid acc = 88.810%, loss = 0.41349397335625904
100%|██████████| 782/782 [00:45<00:00, 17.34it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.135%, loss = 0.05628816136027998
100%|██████████| 79/79 [00:03<00:00, 21.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 34, valid acc = 88.780%, loss = 0.41973642174956166
100%|██████████| 782/782 [00:45<00:00, 17.30it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.266%, loss = 0.053176101274512076
100%|██████████| 79/79 [00:03<00:00, 21.09it/s]
epoch = 35, valid acc = 88.280%, loss = 0.4449241629129724
100%|██████████| 782/782 [00:45<00:00, 17.29it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.197%, loss = 0.052532944303658576
100%|██████████| 79/79 [00:03<00:00, 21.08it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 36, valid acc = 88.170%, loss = 0.4521736624308779
100%|██████████| 782/782 [00:45<00:00, 17.30it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.258%, loss = 0.05034696712227695
100%|██████████| 79/79 [00:03<00:00, 21.45it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 37, valid acc = 88.600%, loss = 0.4413015265253526
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.173%, loss = 0.052666521793150387
100%|██████████| 79/79 [00:03<00:00, 21.07it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 38, valid acc = 88.270%, loss = 0.4448052499490448
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
train acc = 98.282%, loss = 0.050592527854616
100%|██████████| 79/79 [00:03<00:00, 21.15it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 39, valid acc = 88.280%, loss = 0.43930704533299314
100%|██████████| 782/782 [00:45<00:00, 17.27it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.716%, loss = 0.03867798094587672
100%|██████████| 79/79 [00:03<00:00, 21.13it/s]
epoch = 40, valid acc = 88.740%, loss = 0.44620741375639467
100%|██████████| 782/782 [00:45<00:00, 17.29it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 98.926%, loss = 0.03357316389688245
100%|██████████| 79/79 [00:03<00:00, 21.06it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 41, valid acc = 88.740%, loss = 0.45691455429113365
100%|██████████| 782/782 [00:45<00:00, 17.27it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.031%, loss = 0.031187047765059088
100%|██████████| 79/79 [00:03<00:00, 21.08it/s]
epoch = 42, valid acc = 88.850%, loss = 0.436424319125429
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
train acc = 99.064%, loss = 0.029630849756005093
100%|██████████| 79/79 [00:03<00:00, 21.34it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 43, valid acc = 88.930%, loss = 0.4510255656853507
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.072%, loss = 0.0297201087271385
100%|██████████| 79/79 [00:03<00:00, 21.14it/s]
epoch = 44, valid acc = 89.120%, loss = 0.4374161749323712
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.124%, loss = 0.027653961625221705
100%|██████████| 79/79 [00:03<00:00, 21.01it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 45, valid acc = 88.770%, loss = 0.46544227136086813
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.169%, loss = 0.026613044789206248
100%|██████████| 79/79 [00:03<00:00, 21.22it/s]
epoch = 46, valid acc = 88.940%, loss = 0.46782954037189484
100%|██████████| 782/782 [00:45<00:00, 17.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.143%, loss = 0.026869654683443857
100%|██████████| 79/79 [00:03<00:00, 21.13it/s]
epoch = 47, valid acc = 89.160%, loss = 0.4691975871218911
100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.224%, loss = 0.02527395862570304
100%|██████████| 79/79 [00:03<00:00, 21.05it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 48, valid acc = 88.480%, loss = 0.48584367504602743
100%|██████████| 782/782 [00:45<00:00, 17.25it/s]
train acc = 99.242%, loss = 0.024468732762264796
100%|██████████| 79/79 [00:03<00:00, 21.05it/s]
epoch = 49, valid acc = 88.680%, loss = 0.4739708849523641
100%|██████████| 782/782 [00:45<00:00, 17.19it/s]
train acc = 99.349%, loss = 0.021721343099868493
100%|██████████| 79/79 [00:03<00:00, 21.15it/s]
epoch = 50, valid acc = 88.900%, loss = 0.4626697739468345
100%|██████████| 782/782 [00:45<00:00, 17.19it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.421%, loss = 0.019912410801981608
100%|██████████| 79/79 [00:03<00:00, 21.13it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 51, valid acc = 88.970%, loss = 0.46676428710358053
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.427%, loss = 0.01944653815134307
100%|██████████| 79/79 [00:03<00:00, 21.02it/s]
epoch = 52, valid acc = 89.440%, loss = 0.4577470948047276
100%|██████████| 782/782 [00:45<00:00, 17.35it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.453%, loss = 0.018869271927365504
100%|██████████| 79/79 [00:03<00:00, 21.39it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 53, valid acc = 89.260%, loss = 0.46088120058367527
100%|██████████| 782/782 [00:45<00:00, 17.36it/s]
train acc = 99.473%, loss = 0.018416774214443078
100%|██████████| 79/79 [00:03<00:00, 21.39it/s]
epoch = 54, valid acc = 89.230%, loss = 0.4675203295825403
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
train acc = 99.478%, loss = 0.017671101021251696
100%|██████████| 79/79 [00:03<00:00, 21.08it/s]
epoch = 55, valid acc = 89.460%, loss = 0.4616585990673379
100%|██████████| 782/782 [00:45<00:00, 17.15it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.514%, loss = 0.01754077830050579
100%|██████████| 79/79 [00:03<00:00, 20.94it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 56, valid acc = 88.680%, loss = 0.48247432029699977
100%|██████████| 782/782 [00:45<00:00, 17.21it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.505%, loss = 0.01730504244386726
100%|██████████| 79/79 [00:03<00:00, 21.14it/s]
epoch = 57, valid acc = 88.940%, loss = 0.48564893129882936
100%|██████████| 782/782 [00:45<00:00, 17.27it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.544%, loss = 0.016224303295600043
100%|██████████| 79/79 [00:03<00:00, 21.36it/s]
epoch = 58, valid acc = 89.040%, loss = 0.4911715618417233
100%|██████████| 782/782 [00:45<00:00, 17.30it/s]
train acc = 99.501%, loss = 0.017041545547366314
100%|██████████| 79/79 [00:03<00:00, 20.97it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 59, valid acc = 89.190%, loss = 0.4612624062390267
100%|██████████| 782/782 [00:45<00:00, 17.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.578%, loss = 0.015187712244229759
100%|██████████| 79/79 [00:03<00:00, 20.97it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 60, valid acc = 89.170%, loss = 0.47527535429483725
100%|██████████| 782/782 [00:45<00:00, 17.19it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.559%, loss = 0.015542223757041184
100%|██████████| 79/79 [00:03<00:00, 21.02it/s]
epoch = 61, valid acc = 89.240%, loss = 0.468695100349716
100%|██████████| 782/782 [00:45<00:00, 17.14it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.583%, loss = 0.015150329189957417
100%|██████████| 79/79 [00:03<00:00, 20.99it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 62, valid acc = 89.020%, loss = 0.4828618502692331
100%|██████████| 782/782 [00:45<00:00, 17.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.601%, loss = 0.01438598430154564
100%|██████████| 79/79 [00:03<00:00, 20.90it/s]
epoch = 63, valid acc = 88.940%, loss = 0.48123245884346055
100%|██████████| 782/782 [00:45<00:00, 17.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.602%, loss = 0.014287653703229439
100%|██████████| 79/79 [00:03<00:00, 21.20it/s]
epoch = 64, valid acc = 89.440%, loss = 0.46399131681345684
100%|██████████| 782/782 [00:45<00:00, 17.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.643%, loss = 0.013334629873809454
100%|██████████| 79/79 [00:03<00:00, 21.12it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 65, valid acc = 89.580%, loss = 0.47110010721260986
100%|██████████| 782/782 [00:45<00:00, 17.15it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.611%, loss = 0.013747178013627048
100%|██████████| 79/79 [00:03<00:00, 20.98it/s]
epoch = 66, valid acc = 88.890%, loss = 0.47542822530752493
100%|██████████| 782/782 [00:45<00:00, 17.11it/s]
train acc = 99.628%, loss = 0.013402639001118415
100%|██████████| 79/79 [00:03<00:00, 21.09it/s]
epoch = 67, valid acc = 88.640%, loss = 0.5079020254219635
100%|██████████| 782/782 [00:45<00:00, 17.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.646%, loss = 0.013632414207491271
100%|██████████| 79/79 [00:03<00:00, 21.14it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 68, valid acc = 89.320%, loss = 0.4745181342468986
100%|██████████| 782/782 [00:46<00:00, 16.96it/s]
train acc = 99.658%, loss = 0.01307744829027611
100%|██████████| 79/79 [00:03<00:00, 20.31it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 69, valid acc = 88.920%, loss = 0.48563778645630123
100%|██████████| 782/782 [00:46<00:00, 16.98it/s]
train acc = 99.674%, loss = 0.012723225124967534
100%|██████████| 79/79 [00:03<00:00, 20.84it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 70, valid acc = 89.150%, loss = 0.4735279735885089
100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
train acc = 99.637%, loss = 0.013005353630219093
100%|██████████| 79/79 [00:03<00:00, 19.92it/s]
epoch = 71, valid acc = 89.330%, loss = 0.4799186836315107
100%|██████████| 782/782 [00:45<00:00, 17.13it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.644%, loss = 0.012976181192431942
100%|██████████| 79/79 [00:03<00:00, 20.83it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 72, valid acc = 89.160%, loss = 0.48097189959091474
100%|██████████| 782/782 [00:44<00:00, 17.38it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.648%, loss = 0.012644296808107554
100%|██████████| 79/79 [00:03<00:00, 21.14it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 73, valid acc = 88.920%, loss = 0.49278706727148613
100%|██████████| 782/782 [00:44<00:00, 17.47it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.689%, loss = 0.01173782673866257
100%|██████████| 79/79 [00:03<00:00, 21.31it/s]
epoch = 74, valid acc = 89.060%, loss = 0.48751521204845816
100%|██████████| 782/782 [00:45<00:00, 17.34it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.652%, loss = 0.012421662623123468
100%|██████████| 79/79 [00:03<00:00, 21.28it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 75, valid acc = 89.410%, loss = 0.4733183153822452
100%|██████████| 782/782 [00:45<00:00, 17.36it/s]
train acc = 99.715%, loss = 0.011499397299142884
100%|██████████| 79/79 [00:03<00:00, 20.71it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 76, valid acc = 89.260%, loss = 0.4820654982630211
100%|██████████| 782/782 [00:45<00:00, 17.08it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.694%, loss = 0.011702253591761117
100%|██████████| 79/79 [00:03<00:00, 21.04it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 77, valid acc = 89.310%, loss = 0.46224131048480166
100%|██████████| 782/782 [00:45<00:00, 17.17it/s]
train acc = 99.701%, loss = 0.011829683018694668
100%|██████████| 79/79 [00:03<00:00, 20.89it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 78, valid acc = 89.220%, loss = 0.4703483126963241
100%|██████████| 782/782 [00:45<00:00, 17.26it/s]
train acc = 99.733%, loss = 0.01130785132044345
100%|██████████| 79/79 [00:03<00:00, 20.60it/s]
epoch = 79, valid acc = 89.000%, loss = 0.5064855216047431
100%|██████████| 782/782 [00:45<00:00, 17.24it/s]
train acc = 99.691%, loss = 0.012077054301368507
100%|██████████| 79/79 [00:03<00:00, 20.99it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 80, valid acc = 89.470%, loss = 0.4846710864501663
100%|██████████| 782/782 [00:45<00:00, 17.26it/s]
train acc = 99.722%, loss = 0.011423371750843542
100%|██████████| 79/79 [00:03<00:00, 21.61it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 81, valid acc = 88.980%, loss = 0.4959228202889237
100%|██████████| 782/782 [00:45<00:00, 17.32it/s]
train acc = 99.690%, loss = 0.011677018642310015
100%|██████████| 79/79 [00:03<00:00, 20.86it/s]
epoch = 82, valid acc = 88.970%, loss = 0.4829864939556846
100%|██████████| 782/782 [00:45<00:00, 17.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.698%, loss = 0.011655299551546802
100%|██████████| 79/79 [00:03<00:00, 21.08it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 83, valid acc = 89.160%, loss = 0.48666657865801943
100%|██████████| 782/782 [00:45<00:00, 17.09it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.715%, loss = 0.01168271834584539
100%|██████████| 79/79 [00:03<00:00, 21.23it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 84, valid acc = 88.900%, loss = 0.48091133599039876
100%|██████████| 782/782 [00:45<00:00, 17.29it/s]
train acc = 99.752%, loss = 0.01061312082122959
100%|██████████| 79/79 [00:03<00:00, 21.13it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 85, valid acc = 89.110%, loss = 0.48151708648929115
100%|██████████| 782/782 [00:45<00:00, 17.01it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.706%, loss = 0.011459585470969187
100%|██████████| 79/79 [00:03<00:00, 20.99it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 86, valid acc = 89.270%, loss = 0.47691217027133026
100%|██████████| 782/782 [00:46<00:00, 16.83it/s]
train acc = 99.697%, loss = 0.011489581377299078
100%|██████████| 79/79 [00:03<00:00, 20.88it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 87, valid acc = 89.000%, loss = 0.49337522874150097
100%|██████████| 782/782 [00:45<00:00, 17.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.707%, loss = 0.011023575134868341
100%|██████████| 79/79 [00:03<00:00, 20.91it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 88, valid acc = 89.150%, loss = 0.4818977349543873
100%|██████████| 782/782 [00:46<00:00, 16.85it/s]
train acc = 99.748%, loss = 0.010933478361220739
100%|██████████| 79/79 [00:03<00:00, 20.72it/s]
epoch = 89, valid acc = 88.960%, loss = 0.4938147466016721
100%|██████████| 782/782 [00:46<00:00, 16.95it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.700%, loss = 0.011201393524514717
100%|██████████| 79/79 [00:03<00:00, 20.86it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 90, valid acc = 88.860%, loss = 0.49514364385152165
100%|██████████| 782/782 [00:46<00:00, 16.97it/s]
train acc = 99.673%, loss = 0.011694927344216592
100%|██████████| 79/79 [00:03<00:00, 20.86it/s]
epoch = 91, valid acc = 89.040%, loss = 0.4958771733166296
100%|██████████| 782/782 [00:45<00:00, 17.01it/s]
train acc = 99.746%, loss = 0.010959985218626563
100%|██████████| 79/79 [00:03<00:00, 20.71it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 92, valid acc = 89.370%, loss = 0.4815391635970224
100%|██████████| 782/782 [00:45<00:00, 17.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.741%, loss = 0.010505544473988758
100%|██████████| 79/79 [00:03<00:00, 20.87it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 93, valid acc = 89.070%, loss = 0.4965382218360901
100%|██████████| 782/782 [00:46<00:00, 17.00it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.758%, loss = 0.010270455752855496
100%|██████████| 79/79 [00:03<00:00, 20.93it/s]
epoch = 94, valid acc = 89.370%, loss = 0.4620004334027254
100%|██████████| 782/782 [00:45<00:00, 17.20it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.723%, loss = 0.01064076569110579
100%|██████████| 79/79 [00:03<00:00, 21.13it/s]
  0%|          | 0/782 [00:00<?, ?it/s]epoch = 95, valid acc = 89.140%, loss = 0.4871087981553017
100%|██████████| 782/782 [00:45<00:00, 17.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.727%, loss = 0.01082703717442377
100%|██████████| 79/79 [00:03<00:00, 20.83it/s]
epoch = 96, valid acc = 89.550%, loss = 0.46797728123544136
100%|██████████| 782/782 [00:46<00:00, 16.95it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 99.699%, loss = 0.010871644345763595
100%|██████████| 79/79 [00:03<00:00, 19.92it/s]
epoch = 97, valid acc = 89.180%, loss = 0.4838004116016098
100%|██████████| 782/782 [00:46<00:00, 16.83it/s]
train acc = 99.744%, loss = 0.010534192835453593
100%|██████████| 79/79 [00:03<00:00, 21.51it/s]
epoch = 98, valid acc = 89.230%, loss = 0.48050362182946144
```



![image-20240415233416895](D:\Code\SCH\Deit_repetition\assets\image-20240415233416895.png)





```python
# [Pytorch实战2：ResNet-18实现Cifar-10图像分类（测试集分类准确率95.170%）_resnet18 cifar10-CSDN博客](https://blog.csdn.net/sunqiande88/article/details/80100891)

ResNet18
accuracy = 88%

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)




```

![image-20240415224013738](D:\Code\SCH\Deit_repetition\assets\image-20240415224013738.png)





```
"vit2_args": {
    "image_size": 32,
    "patch_size": 2,
    "num_classes": 10,
    "dim": 288,
    "depth": 6,
    "heads": 6,
    "mlp_dim": 576,
    "pool":"cls",
    "channels": 3,
    "dim_head": 48,
    "dropout": 0.5,
    "emb_dropout":0.5
  }
```

![image-20240415195758688](D:\Code\SCH\Deit_repetition\assets\image-20240415195758688.png)



```
"vit_args": {
    "in_channels": 3,
    "patch_size": 2,
    "embed_size": 336,
    "image_size": 32,
    "depth": 5,
    "num_heads": 6,
    "drop_p": 0.3,
    "forward_expansion": 4,
    "forward_drop_p": 0.2,
    "num_classes": 10,
    "if_cls": true,
    "if_dis": false
  }
```

![image-20240415204214065](D:\Code\SCH\Deit_repetition\assets\image-20240415204214065.png)







```
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 18.720%, loss = 2.1265540470552566
100%|██████████| 79/79 [00:09<00:00,  8.25it/s]
epoch = 0, valid acc = 26.200%, loss = 1.9646647051919865
100%|██████████| 391/391 [02:01<00:00,  3.21it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 27.108%, loss = 1.936747168336073
100%|██████████| 79/79 [00:09<00:00,  8.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 1, valid acc = 31.560%, loss = 1.847146883795533
100%|██████████| 391/391 [02:02<00:00,  3.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 32.916%, loss = 1.8113414372324639
100%|██████████| 79/79 [00:09<00:00,  8.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 2, valid acc = 38.090%, loss = 1.7227842626692373
100%|██████████| 391/391 [02:02<00:00,  3.19it/s]
train acc = 36.758%, loss = 1.7199988587737998
100%|██████████| 79/79 [00:09<00:00,  8.31it/s]
epoch = 3, valid acc = 39.830%, loss = 1.6573469910440566
100%|██████████| 391/391 [02:03<00:00,  3.17it/s]
train acc = 39.642%, loss = 1.6515281691270716
100%|██████████| 79/79 [00:09<00:00,  8.42it/s]
epoch = 4, valid acc = 43.550%, loss = 1.561707421194149
100%|██████████| 391/391 [02:02<00:00,  3.19it/s]
train acc = 42.220%, loss = 1.5810243757179632
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 5, valid acc = 46.370%, loss = 1.4838671186302281
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 44.860%, loss = 1.5111578483410808
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 6, valid acc = 47.510%, loss = 1.4400181800504275
100%|██████████| 391/391 [02:05<00:00,  3.13it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 46.700%, loss = 1.4611379493532888
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 7, valid acc = 49.800%, loss = 1.3978173249884496
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 48.648%, loss = 1.418687773482574
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
epoch = 8, valid acc = 50.750%, loss = 1.3684082348135453
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
train acc = 49.488%, loss = 1.393024463787713
100%|██████████| 79/79 [00:09<00:00,  7.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 9, valid acc = 51.460%, loss = 1.3447763557675518
100%|██████████| 391/391 [02:07<00:00,  3.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 50.670%, loss = 1.363961866139756
100%|██████████| 79/79 [00:09<00:00,  8.44it/s]
epoch = 10, valid acc = 52.240%, loss = 1.320822303808188
100%|██████████| 391/391 [02:00<00:00,  3.24it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 51.182%, loss = 1.338792069183896
100%|██████████| 79/79 [00:09<00:00,  8.63it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 11, valid acc = 52.570%, loss = 1.2997212470332278
100%|██████████| 391/391 [02:00<00:00,  3.24it/s]
train acc = 51.940%, loss = 1.3229374287988218
100%|██████████| 79/79 [00:09<00:00,  8.63it/s]
epoch = 12, valid acc = 54.030%, loss = 1.2716507496713083
100%|██████████| 391/391 [02:02<00:00,  3.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.516%, loss = 1.3059970071858458
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 13, valid acc = 54.100%, loss = 1.2659238878684709
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
train acc = 53.274%, loss = 1.289965056092538
100%|██████████| 79/79 [00:09<00:00,  8.62it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 14, valid acc = 54.810%, loss = 1.2567080380041389
100%|██████████| 391/391 [02:00<00:00,  3.23it/s]
train acc = 53.648%, loss = 1.2784992325336426
100%|██████████| 79/79 [00:09<00:00,  8.50it/s]
epoch = 15, valid acc = 55.340%, loss = 1.2429907533186901
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
train acc = 53.824%, loss = 1.2690095200258142
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 16, valid acc = 55.320%, loss = 1.24215160442304
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 54.184%, loss = 1.260892286629933
100%|██████████| 79/79 [00:09<00:00,  8.55it/s]
epoch = 17, valid acc = 55.590%, loss = 1.2358697019045866
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.504%, loss = 1.2554455873606456
100%|██████████| 79/79 [00:09<00:00,  8.30it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 18, valid acc = 55.970%, loss = 1.2248683013493502
100%|██████████| 391/391 [02:03<00:00,  3.18it/s]
train acc = 54.804%, loss = 1.249127058879189
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 19, valid acc = 55.370%, loss = 1.2268414648273323
100%|██████████| 391/391 [02:04<00:00,  3.14it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.714%, loss = 1.2496122484621794
100%|██████████| 79/79 [00:09<00:00,  8.45it/s]
epoch = 20, valid acc = 55.580%, loss = 1.2239581844474696
100%|██████████| 391/391 [02:04<00:00,  3.14it/s]
train acc = 54.768%, loss = 1.2504920364950625
100%|██████████| 79/79 [00:09<00:00,  8.63it/s]
epoch = 21, valid acc = 55.710%, loss = 1.2311839438691925
100%|██████████| 391/391 [02:06<00:00,  3.08it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.508%, loss = 1.251099022300652
100%|██████████| 79/79 [00:09<00:00,  8.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 22, valid acc = 55.560%, loss = 1.230685540392429
100%|██████████| 391/391 [02:08<00:00,  3.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.692%, loss = 1.2489617852603687
100%|██████████| 79/79 [00:09<00:00,  8.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 23, valid acc = 55.550%, loss = 1.2284680393677723
100%|██████████| 391/391 [02:07<00:00,  3.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.254%, loss = 1.2568599314945739
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
epoch = 24, valid acc = 55.530%, loss = 1.2324176936209956
100%|██████████| 391/391 [02:03<00:00,  3.18it/s]
train acc = 54.478%, loss = 1.255756269178122
100%|██████████| 79/79 [00:09<00:00,  8.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 25, valid acc = 55.480%, loss = 1.2369323727450794
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
train acc = 54.670%, loss = 1.253613048349805
100%|██████████| 79/79 [00:09<00:00,  8.22it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 26, valid acc = 55.570%, loss = 1.231489507457878
100%|██████████| 391/391 [02:02<00:00,  3.18it/s]
train acc = 54.490%, loss = 1.2525042170453864
100%|██████████| 79/79 [00:09<00:00,  8.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 27, valid acc = 56.100%, loss = 1.2130342246610908
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.696%, loss = 1.2506854887813559
100%|██████████| 79/79 [00:09<00:00,  8.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 28, valid acc = 56.260%, loss = 1.2262129987342447
100%|██████████| 391/391 [02:06<00:00,  3.10it/s]
train acc = 54.526%, loss = 1.2532562925992414
100%|██████████| 79/79 [00:11<00:00,  6.81it/s]
epoch = 29, valid acc = 54.850%, loss = 1.2370858780945404
100%|██████████| 391/391 [02:31<00:00,  2.59it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.810%, loss = 1.243750302383052
100%|██████████| 79/79 [00:09<00:00,  8.30it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 30, valid acc = 55.950%, loss = 1.2411730395087712
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.996%, loss = 1.23806793000692
100%|██████████| 79/79 [00:09<00:00,  8.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 31, valid acc = 56.800%, loss = 1.2037777674349048
100%|██████████| 391/391 [02:01<00:00,  3.21it/s]
train acc = 55.128%, loss = 1.2390436030104948
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
epoch = 32, valid acc = 57.120%, loss = 1.199059996423842
100%|██████████| 391/391 [02:04<00:00,  3.13it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 55.546%, loss = 1.230900646780458
100%|██████████| 79/79 [00:09<00:00,  8.63it/s]
epoch = 33, valid acc = 57.410%, loss = 1.1834922636611551
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 55.738%, loss = 1.2233446373049255
100%|██████████| 79/79 [00:09<00:00,  8.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 34, valid acc = 57.260%, loss = 1.1953691055503073
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 56.468%, loss = 1.2100679520755777
100%|██████████| 79/79 [00:09<00:00,  8.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 35, valid acc = 53.170%, loss = 1.2897075127951707
100%|██████████| 391/391 [02:04<00:00,  3.15it/s]
train acc = 56.348%, loss = 1.2080114899998735
100%|██████████| 79/79 [00:09<00:00,  8.40it/s]
epoch = 36, valid acc = 57.580%, loss = 1.1772293025934244
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.600%, loss = 1.2025038356061482
100%|██████████| 79/79 [00:09<00:00,  8.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 37, valid acc = 57.820%, loss = 1.161452586892285
100%|██████████| 391/391 [02:04<00:00,  3.15it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.884%, loss = 1.1918721929230653
100%|██████████| 79/79 [00:09<00:00,  8.28it/s]
epoch = 38, valid acc = 57.460%, loss = 1.182254978373081
100%|██████████| 391/391 [02:12<00:00,  2.94it/s]
train acc = 57.334%, loss = 1.1768327492582218
100%|██████████| 79/79 [00:10<00:00,  7.66it/s]
epoch = 39, valid acc = 58.400%, loss = 1.1734767699543434
100%|██████████| 391/391 [02:07<00:00,  3.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 57.858%, loss = 1.1713343114804124
100%|██████████| 79/79 [00:09<00:00,  8.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 40, valid acc = 58.380%, loss = 1.1567989414251303
100%|██████████| 391/391 [02:10<00:00,  2.98it/s]
train acc = 58.296%, loss = 1.1584552800868784
100%|██████████| 79/79 [00:10<00:00,  7.87it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 41, valid acc = 59.300%, loss = 1.1405271819875211
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
train acc = 58.800%, loss = 1.1440047228427799
100%|██████████| 79/79 [00:09<00:00,  8.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 42, valid acc = 58.380%, loss = 1.1670068396797664
100%|██████████| 391/391 [02:06<00:00,  3.08it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.540%, loss = 1.1308064840333847
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
epoch = 43, valid acc = 59.830%, loss = 1.1325160189519954
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.696%, loss = 1.121491419537293
100%|██████████| 79/79 [00:10<00:00,  7.80it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 44, valid acc = 61.050%, loss = 1.0797486342961276
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
train acc = 60.132%, loss = 1.1086014149438999
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 45, valid acc = 60.860%, loss = 1.108241652386098
100%|██████████| 391/391 [02:02<00:00,  3.19it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 60.522%, loss = 1.0957164922943505
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
epoch = 46, valid acc = 61.770%, loss = 1.0801779906960982
100%|██████████| 391/391 [02:09<00:00,  3.03it/s]
train acc = 61.120%, loss = 1.0835097767507937
100%|██████████| 79/79 [00:10<00:00,  7.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 47, valid acc = 60.780%, loss = 1.0969927265674253
100%|██████████| 391/391 [02:01<00:00,  3.21it/s]
train acc = 61.626%, loss = 1.0682521259693234
100%|██████████| 79/79 [00:09<00:00,  8.27it/s]
epoch = 48, valid acc = 61.750%, loss = 1.073333268678641
100%|██████████| 391/391 [02:01<00:00,  3.21it/s]
train acc = 62.340%, loss = 1.0529959634746737
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 49, valid acc = 62.210%, loss = 1.0599940057042279
100%|██████████| 391/391 [02:07<00:00,  3.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 62.580%, loss = 1.0356090279186474
100%|██████████| 79/79 [00:09<00:00,  8.34it/s]
epoch = 50, valid acc = 62.800%, loss = 1.0366127445727964
100%|██████████| 391/391 [02:03<00:00,  3.18it/s]
train acc = 63.022%, loss = 1.0280480753735204
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
epoch = 51, valid acc = 63.320%, loss = 1.0377177467829064
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 63.932%, loss = 1.0088376055288193
100%|██████████| 79/79 [00:10<00:00,  7.85it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 52, valid acc = 63.660%, loss = 1.0307958005349847
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 63.992%, loss = 1.0040630689057548
100%|██████████| 79/79 [00:09<00:00,  8.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 53, valid acc = 63.760%, loss = 1.0311664931381805
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.314%, loss = 0.9894763363901612
100%|██████████| 79/79 [00:09<00:00,  8.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 54, valid acc = 64.100%, loss = 1.0166559611694723
100%|██████████| 391/391 [02:15<00:00,  2.89it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.536%, loss = 0.9839874374896974
100%|██████████| 79/79 [00:09<00:00,  8.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 55, valid acc = 64.130%, loss = 1.014588419395157
100%|██████████| 391/391 [02:13<00:00,  2.92it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.148%, loss = 0.9739639574609449
100%|██████████| 79/79 [00:10<00:00,  7.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 56, valid acc = 64.250%, loss = 1.0081493846977814
100%|██████████| 391/391 [02:06<00:00,  3.10it/s]
train acc = 65.086%, loss = 0.9679641098622471
100%|██████████| 79/79 [00:10<00:00,  7.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 57, valid acc = 64.460%, loss = 1.0062301619143426
100%|██████████| 391/391 [02:13<00:00,  2.93it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.418%, loss = 0.9620751081525213
100%|██████████| 79/79 [00:10<00:00,  7.65it/s]
epoch = 58, valid acc = 64.420%, loss = 0.9998802874661699
100%|██████████| 391/391 [02:11<00:00,  2.98it/s]
train acc = 65.552%, loss = 0.9612363397008012
100%|██████████| 79/79 [00:10<00:00,  7.90it/s]
epoch = 59, valid acc = 64.910%, loss = 0.9956165061721319
100%|██████████| 391/391 [02:09<00:00,  3.01it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.478%, loss = 0.9612187009943111
100%|██████████| 79/79 [00:10<00:00,  7.87it/s]
epoch = 60, valid acc = 64.390%, loss = 1.0008926534954505
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
train acc = 65.312%, loss = 0.9626422893360752
100%|██████████| 79/79 [00:09<00:00,  8.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 61, valid acc = 65.260%, loss = 0.987871285480789
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 65.558%, loss = 0.9624819093957886
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 62, valid acc = 64.700%, loss = 0.9991319594503958
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.442%, loss = 0.964720606803894
100%|██████████| 79/79 [00:10<00:00,  7.87it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 63, valid acc = 64.930%, loss = 0.9976790690723853
100%|██████████| 391/391 [02:07<00:00,  3.06it/s]
train acc = 65.316%, loss = 0.9669980057670028
100%|██████████| 79/79 [00:10<00:00,  7.61it/s]
epoch = 64, valid acc = 64.990%, loss = 0.9962200190447554
100%|██████████| 391/391 [02:04<00:00,  3.14it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.180%, loss = 0.9740135898370572
100%|██████████| 79/79 [00:09<00:00,  8.18it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 65, valid acc = 64.180%, loss = 1.0057723499551605
100%|██████████| 391/391 [02:07<00:00,  3.08it/s]
train acc = 65.198%, loss = 0.9698784331531476
100%|██████████| 79/79 [00:10<00:00,  7.87it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 66, valid acc = 64.180%, loss = 1.014391908917246
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.018%, loss = 0.9799233466158133
100%|██████████| 79/79 [00:09<00:00,  8.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 67, valid acc = 64.090%, loss = 1.0308781620822376
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.826%, loss = 0.9835574758022337
100%|██████████| 79/79 [00:09<00:00,  8.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 68, valid acc = 64.610%, loss = 0.9962073836145522
100%|██████████| 391/391 [02:06<00:00,  3.10it/s]
train acc = 64.802%, loss = 0.9856769884638774
100%|██████████| 79/79 [00:09<00:00,  8.22it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 69, valid acc = 64.280%, loss = 1.0029331249526785
100%|██████████| 391/391 [02:13<00:00,  2.93it/s]
train acc = 64.590%, loss = 0.9870371955739873
100%|██████████| 79/79 [00:09<00:00,  8.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 70, valid acc = 64.750%, loss = 0.9941971626462816
100%|██████████| 391/391 [02:07<00:00,  3.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.476%, loss = 0.99050099541769
100%|██████████| 79/79 [00:09<00:00,  7.91it/s]
epoch = 71, valid acc = 64.100%, loss = 1.0340477152715755
100%|██████████| 391/391 [02:06<00:00,  3.10it/s]
train acc = 64.506%, loss = 0.9892699727621834
100%|██████████| 79/79 [00:09<00:00,  8.36it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 72, valid acc = 63.910%, loss = 1.0169219087950792
100%|██████████| 391/391 [02:07<00:00,  3.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.236%, loss = 0.9930190690947921
100%|██████████| 79/79 [00:09<00:00,  8.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 73, valid acc = 63.530%, loss = 1.042316725736932
100%|██████████| 391/391 [02:03<00:00,  3.15it/s]
train acc = 64.452%, loss = 0.99298295477772
100%|██████████| 79/79 [00:09<00:00,  8.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 74, valid acc = 64.470%, loss = 1.000427937960323
100%|██████████| 391/391 [02:04<00:00,  3.14it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.720%, loss = 0.9896354608218688
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 75, valid acc = 63.780%, loss = 1.0189437194715572
100%|██████████| 391/391 [02:02<00:00,  3.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.790%, loss = 0.988622076980903
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 76, valid acc = 64.470%, loss = 1.0108338194557382
100%|██████████| 391/391 [02:01<00:00,  3.21it/s]
train acc = 64.638%, loss = 0.9880588780278745
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 77, valid acc = 64.680%, loss = 0.9999212909348404
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
train acc = 64.930%, loss = 0.9843091138488497
100%|██████████| 79/79 [00:09<00:00,  8.54it/s]
epoch = 78, valid acc = 64.260%, loss = 1.0083246065091482
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 64.800%, loss = 0.9778492356200352
100%|██████████| 79/79 [00:09<00:00,  8.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 79, valid acc = 64.940%, loss = 0.9865788794771025
100%|██████████| 391/391 [02:13<00:00,  2.94it/s]
train acc = 64.790%, loss = 0.9764835166809199
100%|██████████| 79/79 [00:09<00:00,  8.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 80, valid acc = 65.580%, loss = 0.9742474427706078
100%|██████████| 391/391 [02:10<00:00,  3.00it/s]
train acc = 65.382%, loss = 0.9626809137556559
100%|██████████| 79/79 [00:10<00:00,  7.75it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 81, valid acc = 64.590%, loss = 1.0098345317418063
100%|██████████| 391/391 [02:04<00:00,  3.15it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.986%, loss = 0.954185907493162
100%|██████████| 79/79 [00:09<00:00,  8.30it/s]
epoch = 82, valid acc = 65.470%, loss = 0.9841690395451799
100%|██████████| 391/391 [02:04<00:00,  3.15it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.264%, loss = 0.9462142842809868
100%|██████████| 79/79 [00:09<00:00,  8.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 83, valid acc = 66.020%, loss = 0.9563003235225436
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 66.446%, loss = 0.9364149991203757
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 84, valid acc = 66.490%, loss = 0.9521922921832604
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.846%, loss = 0.9265628146088641
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 85, valid acc = 67.370%, loss = 0.9396720189082471
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
train acc = 67.416%, loss = 0.91584118857713
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 86, valid acc = 66.680%, loss = 0.9336281026465983
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
train acc = 67.794%, loss = 0.9051021424400837
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 87, valid acc = 67.270%, loss = 0.9123468625394604
100%|██████████| 391/391 [02:06<00:00,  3.08it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.144%, loss = 0.8928816605102071
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
epoch = 88, valid acc = 67.830%, loss = 0.9168037892896918
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
train acc = 68.774%, loss = 0.8786997618272786
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 89, valid acc = 68.540%, loss = 0.9163310241095627
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
train acc = 68.938%, loss = 0.8668239875827604
100%|██████████| 79/79 [00:10<00:00,  7.70it/s]
epoch = 90, valid acc = 68.040%, loss = 0.9066249245329748
100%|██████████| 391/391 [02:12<00:00,  2.96it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.446%, loss = 0.8578570178707542
100%|██████████| 79/79 [00:09<00:00,  8.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 91, valid acc = 69.080%, loss = 0.8809812747979466
100%|██████████| 391/391 [02:05<00:00,  3.11it/s]
train acc = 70.038%, loss = 0.8414003984702517
100%|██████████| 79/79 [00:09<00:00,  8.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 92, valid acc = 69.360%, loss = 0.8820327165760572
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.236%, loss = 0.8325583436300078
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 93, valid acc = 69.810%, loss = 0.8709095482584797
100%|██████████| 391/391 [02:02<00:00,  3.19it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.866%, loss = 0.822371953741059
100%|██████████| 79/79 [00:09<00:00,  8.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 94, valid acc = 69.580%, loss = 0.8847879118557218
100%|██████████| 391/391 [02:08<00:00,  3.04it/s]
train acc = 70.964%, loss = 0.8117773706650795
100%|██████████| 79/79 [00:09<00:00,  8.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 95, valid acc = 69.720%, loss = 0.8671398985234997
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 71.422%, loss = 0.8041700313768119
100%|██████████| 79/79 [00:09<00:00,  8.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 96, valid acc = 69.670%, loss = 0.8792861451076556
100%|██████████| 391/391 [02:14<00:00,  2.91it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.448%, loss = 0.7964249454495852
100%|██████████| 79/79 [00:09<00:00,  8.55it/s]
epoch = 97, valid acc = 69.720%, loss = 0.8740407160565823
100%|██████████| 391/391 [02:09<00:00,  3.01it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.616%, loss = 0.7957632860259327
100%|██████████| 79/79 [00:09<00:00,  8.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 98, valid acc = 70.280%, loss = 0.8625199953212014
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 71.744%, loss = 0.7884569338825352
100%|██████████| 79/79 [00:09<00:00,  8.31it/s]
epoch = 99, valid acc = 69.510%, loss = 0.8738403101510639
100%|██████████| 391/391 [02:06<00:00,  3.09it/s]
train acc = 71.822%, loss = 0.7875353200051486
100%|██████████| 79/79 [00:09<00:00,  8.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 100, valid acc = 70.070%, loss = 0.8655394949490511
100%|██████████| 391/391 [02:08<00:00,  3.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.064%, loss = 0.787406870166359
100%|██████████| 79/79 [00:10<00:00,  7.86it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 101, valid acc = 69.980%, loss = 0.8715627925305427
100%|██████████| 391/391 [02:04<00:00,  3.14it/s]
train acc = 71.866%, loss = 0.7902640829915586
100%|██████████| 79/79 [00:11<00:00,  7.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 102, valid acc = 69.430%, loss = 0.8748889218402814
100%|██████████| 391/391 [02:13<00:00,  2.93it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.856%, loss = 0.7877689109129065
100%|██████████| 79/79 [00:10<00:00,  7.76it/s]
epoch = 103, valid acc = 70.060%, loss = 0.8727482504482511
100%|██████████| 391/391 [02:09<00:00,  3.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.852%, loss = 0.7959903757590467
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 104, valid acc = 70.450%, loss = 0.859715773334986
100%|██████████| 391/391 [02:11<00:00,  2.98it/s]
train acc = 71.426%, loss = 0.8014284736665008
100%|██████████| 79/79 [00:09<00:00,  8.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 105, valid acc = 70.170%, loss = 0.866630109805095
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.346%, loss = 0.8074167518664503
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
epoch = 106, valid acc = 70.180%, loss = 0.859107692784901
100%|██████████| 391/391 [02:10<00:00,  3.00it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.128%, loss = 0.8078291570134175
100%|██████████| 79/79 [00:11<00:00,  7.18it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 107, valid acc = 70.320%, loss = 0.8497202464296848
100%|██████████| 391/391 [02:11<00:00,  2.98it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.960%, loss = 0.8187777595141964
100%|██████████| 79/79 [00:09<00:00,  8.24it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 108, valid acc = 70.120%, loss = 0.8750436464442483
100%|██████████| 391/391 [02:05<00:00,  3.12it/s]
train acc = 70.630%, loss = 0.8217451066312278
100%|██████████| 79/79 [00:10<00:00,  7.24it/s]
epoch = 109, valid acc = 68.860%, loss = 0.8935513375680658
100%|██████████| 391/391 [02:08<00:00,  3.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.574%, loss = 0.8299781742608151
100%|██████████| 79/79 [00:09<00:00,  8.30it/s]
epoch = 110, valid acc = 69.700%, loss = 0.8677036799962008
100%|██████████| 391/391 [02:03<00:00,  3.18it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.448%, loss = 0.8299681512291169
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 111, valid acc = 68.950%, loss = 0.8953599809091303
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.078%, loss = 0.8376129504359896
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 112, valid acc = 68.420%, loss = 0.8919195336631581
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.310%, loss = 0.8373077252636785
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 113, valid acc = 69.110%, loss = 0.9012871067735213
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
train acc = 69.780%, loss = 0.8462768883046592
100%|██████████| 79/79 [00:09<00:00,  8.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 114, valid acc = 68.700%, loss = 0.8947141872176642
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.772%, loss = 0.8482194003241751
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 115, valid acc = 68.350%, loss = 0.8902406579331507
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.758%, loss = 0.8493823939577088
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 116, valid acc = 67.570%, loss = 0.9379042258745507
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.746%, loss = 0.8507321862613454
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 117, valid acc = 67.850%, loss = 0.9260753114012223
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.788%, loss = 0.8473475156232829
100%|██████████| 79/79 [00:09<00:00,  8.54it/s]
epoch = 118, valid acc = 68.890%, loss = 0.9114787880378433
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 69.672%, loss = 0.845146070813279
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 119, valid acc = 68.820%, loss = 0.8952953475940076
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 70.340%, loss = 0.835713066865721
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
epoch = 120, valid acc = 68.710%, loss = 0.9029849963852122
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 70.318%, loss = 0.8381847914527444
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 121, valid acc = 69.600%, loss = 0.8792832855936847
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.534%, loss = 0.831501543216998
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 122, valid acc = 69.300%, loss = 0.8826942700373975
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.684%, loss = 0.8270536697734042
100%|██████████| 79/79 [00:09<00:00,  8.55it/s]
epoch = 123, valid acc = 69.130%, loss = 0.891206266004828
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.626%, loss = 0.808229544583489
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
epoch = 124, valid acc = 70.960%, loss = 0.8411862789830075
100%|██████████| 391/391 [02:01<00:00,  3.23it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.436%, loss = 0.8083361881163419
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 125, valid acc = 69.550%, loss = 0.8583643677868421
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
train acc = 71.742%, loss = 0.7957169245880888
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 126, valid acc = 70.880%, loss = 0.8537018382096593
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.234%, loss = 0.7837987781485634
100%|██████████| 79/79 [00:10<00:00,  7.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 127, valid acc = 70.310%, loss = 0.8582126098343089
100%|██████████| 391/391 [02:10<00:00,  3.00it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.444%, loss = 0.7781314695887553
100%|██████████| 79/79 [00:09<00:00,  8.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 128, valid acc = 70.070%, loss = 0.8645535111427307
100%|██████████| 391/391 [02:07<00:00,  3.07it/s]
train acc = 72.834%, loss = 0.7638939220429687
100%|██████████| 79/79 [00:10<00:00,  7.21it/s]
epoch = 129, valid acc = 71.010%, loss = 0.855975524534153
100%|██████████| 391/391 [02:17<00:00,  2.85it/s]
train acc = 73.166%, loss = 0.7523073434372387
100%|██████████| 79/79 [00:09<00:00,  8.15it/s]
epoch = 130, valid acc = 71.790%, loss = 0.8264754271205468
100%|██████████| 391/391 [02:08<00:00,  3.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 73.626%, loss = 0.7454151175820919
100%|██████████| 79/79 [00:09<00:00,  8.22it/s]
epoch = 131, valid acc = 71.240%, loss = 0.8392059222052369
100%|██████████| 391/391 [02:11<00:00,  2.98it/s]
train acc = 73.888%, loss = 0.7263699736436615
100%|██████████| 79/79 [00:10<00:00,  7.23it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 132, valid acc = 72.030%, loss = 0.8134546317631686
100%|██████████| 391/391 [02:06<00:00,  3.08it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.146%, loss = 0.7207044105395637
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 133, valid acc = 72.200%, loss = 0.8084825367867192
100%|██████████| 391/391 [02:15<00:00,  2.88it/s]
train acc = 74.856%, loss = 0.7072323573672253
100%|██████████| 79/79 [00:11<00:00,  7.17it/s]
epoch = 134, valid acc = 72.610%, loss = 0.8209065553508227
100%|██████████| 391/391 [02:03<00:00,  3.16it/s]
train acc = 75.080%, loss = 0.7036735250059601
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 135, valid acc = 72.680%, loss = 0.7906728756578663
100%|██████████| 391/391 [02:01<00:00,  3.22it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.318%, loss = 0.6931019574784867
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 136, valid acc = 72.100%, loss = 0.8051436920709247
100%|██████████| 391/391 [02:02<00:00,  3.18it/s]
train acc = 75.562%, loss = 0.6895415039775926
100%|██████████| 79/79 [00:09<00:00,  8.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 137, valid acc = 72.630%, loss = 0.8054324826107749

```









```
distill result
100%|██████████| 391/391 [02:23<00:00,  2.73it/s]
train acc = 18.280%, loss = 5.151125937166726
100%|██████████| 79/79 [00:09<00:00,  8.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 0, valid acc = 23.240%, loss = 2.0564917763577233
100%|██████████| 391/391 [02:20<00:00,  2.77it/s]
train acc = 26.090%, loss = 4.733635387762123
100%|██████████| 79/79 [00:09<00:00,  8.62it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 1, valid acc = 30.090%, loss = 1.9193034805829012
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 31.992%, loss = 4.447887340774926
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
epoch = 2, valid acc = 36.170%, loss = 1.7839444060868854
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 36.672%, loss = 4.154210211980678
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 3, valid acc = 41.450%, loss = 1.7652854919433594
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 41.028%, loss = 3.923960929636455
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 4, valid acc = 44.670%, loss = 1.632283017605166
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 43.670%, loss = 3.7642746709496775
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 5, valid acc = 46.140%, loss = 1.5803767186176927
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 45.818%, loss = 3.61406048728377
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 6, valid acc = 49.420%, loss = 1.5117241566694235
100%|██████████| 391/391 [02:21<00:00,  2.75it/s]
train acc = 47.774%, loss = 3.5010660133703286
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 7, valid acc = 50.410%, loss = 1.5119211251222635
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 49.562%, loss = 3.395235725071119
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 8, valid acc = 52.000%, loss = 1.4955794826338562
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 51.206%, loss = 3.3006711347633617
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 9, valid acc = 53.090%, loss = 1.4610734197157849
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 51.878%, loss = 3.2357665955867914
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 10, valid acc = 53.200%, loss = 1.483476174028614
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 52.812%, loss = 3.1830979893579507
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 11, valid acc = 54.780%, loss = 1.364786040179337
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 54.170%, loss = 3.1151923035721643
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 12, valid acc = 55.470%, loss = 1.3791269697720492
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.672%, loss = 3.0720375230550157
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 13, valid acc = 56.190%, loss = 1.357761692397202
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 55.280%, loss = 3.0264151004879065
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 14, valid acc = 56.900%, loss = 1.3345437004596372
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 55.954%, loss = 3.009374081936029
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 15, valid acc = 56.640%, loss = 1.3260815196399447
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 56.020%, loss = 2.9889800585139437
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 16, valid acc = 57.040%, loss = 1.3531958257095724
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.604%, loss = 2.96170162361906
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 17, valid acc = 57.510%, loss = 1.3632771108723893
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.958%, loss = 2.9349304745569254
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 18, valid acc = 57.730%, loss = 1.3554978129229969
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 57.062%, loss = 2.941408457353597
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 19, valid acc = 57.600%, loss = 1.3553346221960043
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.772%, loss = 2.939161667128658
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 20, valid acc = 57.760%, loss = 1.3436348800417743
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.952%, loss = 2.941763266882933
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 21, valid acc = 57.390%, loss = 1.3479183595391768
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.952%, loss = 2.9394315709848233
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 22, valid acc = 57.490%, loss = 1.3437769752514512
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.844%, loss = 2.9324667283031336
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 23, valid acc = 57.700%, loss = 1.3295179107521153
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 56.922%, loss = 2.943129799250142
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 24, valid acc = 57.940%, loss = 1.34400359890129
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 56.828%, loss = 2.9372790201240795
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 25, valid acc = 57.730%, loss = 1.3542738715304603
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 57.310%, loss = 2.9283220572849675
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 26, valid acc = 58.040%, loss = 1.3353205647649644
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.768%, loss = 2.9369941434591933
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 27, valid acc = 56.350%, loss = 1.3457252466225926
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 56.952%, loss = 2.9293498773404094
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 28, valid acc = 58.360%, loss = 1.3259329554400867
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 57.342%, loss = 2.920119869739503
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 29, valid acc = 58.470%, loss = 1.3234244071984593
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 57.524%, loss = 2.9146643575195155
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 30, valid acc = 59.170%, loss = 1.3126968960218792
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 57.670%, loss = 2.8973330433106486
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 31, valid acc = 57.530%, loss = 1.420469019986406
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 57.680%, loss = 2.8918840604669906
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 32, valid acc = 58.220%, loss = 1.338449798052824
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 58.204%, loss = 2.8625976472254604
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 33, valid acc = 60.180%, loss = 1.287187648724906
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 58.744%, loss = 2.831495524062525
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 34, valid acc = 59.660%, loss = 1.334698875493641
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 58.916%, loss = 2.8177911991353537
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 35, valid acc = 60.690%, loss = 1.2763904167127005
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 59.296%, loss = 2.797466630521028
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 36, valid acc = 61.140%, loss = 1.2886128463322604
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 59.566%, loss = 2.766891080705101
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 37, valid acc = 60.290%, loss = 1.3300587602808505
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 59.812%, loss = 2.7513460991022836
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 38, valid acc = 61.580%, loss = 1.2614296018322813
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 60.584%, loss = 2.7161956943209518
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
epoch = 39, valid acc = 61.400%, loss = 1.2509047645556777
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 61.372%, loss = 2.6706085461179923
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 40, valid acc = 62.660%, loss = 1.2802802429923528
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 61.118%, loss = 2.65865781026728
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 41, valid acc = 61.790%, loss = 1.2977527991125855
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 61.980%, loss = 2.6134697006791447
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 42, valid acc = 62.100%, loss = 1.3099114736424218
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 62.182%, loss = 2.5974109441118167
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 43, valid acc = 62.750%, loss = 1.293655804440945
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 62.906%, loss = 2.553797764241543
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 44, valid acc = 62.920%, loss = 1.2143018479588665
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 63.610%, loss = 2.520215668641698
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 45, valid acc = 64.420%, loss = 1.1848120417775987
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 63.878%, loss = 2.499257063011989
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 46, valid acc = 64.010%, loss = 1.2391551318047922
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 64.534%, loss = 2.4596561089805933
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 47, valid acc = 63.770%, loss = 1.2476691664019717
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 64.866%, loss = 2.429282875012254
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 48, valid acc = 65.070%, loss = 1.2418716270712358
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 65.616%, loss = 2.38794429222946
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 49, valid acc = 64.090%, loss = 1.26973867265484
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.882%, loss = 2.3607526795028724
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 50, valid acc = 65.630%, loss = 1.1630727331849593
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.358%, loss = 2.326003938989566
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 51, valid acc = 66.430%, loss = 1.1745838402192803
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.900%, loss = 2.294769756019573
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 52, valid acc = 66.800%, loss = 1.1733233981494662
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.404%, loss = 2.2723415724151885
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 53, valid acc = 66.430%, loss = 1.1667080023620702
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.642%, loss = 2.240497248556913
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 54, valid acc = 66.640%, loss = 1.1784356485439251
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.160%, loss = 2.2137766905757776
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 55, valid acc = 67.420%, loss = 1.1574146370344525
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.240%, loss = 2.2037714792944283
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 56, valid acc = 67.220%, loss = 1.1549164781087562
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 68.434%, loss = 2.185477587877942
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 57, valid acc = 67.960%, loss = 1.1797115893303594
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.734%, loss = 2.1731459935912696
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 58, valid acc = 67.690%, loss = 1.150418152537527
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 68.778%, loss = 2.1717381992608384
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 59, valid acc = 67.850%, loss = 1.1566170348396785
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.910%, loss = 2.1688803291076892
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 60, valid acc = 67.560%, loss = 1.1636789139313033
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.876%, loss = 2.162979979344341
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 61, valid acc = 67.740%, loss = 1.184409633467469
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 68.656%, loss = 2.1774829619985714
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 62, valid acc = 67.960%, loss = 1.172217499606217
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 68.688%, loss = 2.1787601544728976
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 63, valid acc = 67.490%, loss = 1.1779186499269703
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.704%, loss = 2.186395450626188
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 64, valid acc = 67.650%, loss = 1.1859214871744566
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.374%, loss = 2.1954329827862322
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 65, valid acc = 66.910%, loss = 1.1901742416092111
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.256%, loss = 2.1974444663738044
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 66, valid acc = 67.500%, loss = 1.1910905528672133
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 68.242%, loss = 2.210064231587188
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 67, valid acc = 67.960%, loss = 1.1524366230904302
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.050%, loss = 2.2244766868288863
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 68, valid acc = 67.010%, loss = 1.2222180607952648
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.808%, loss = 2.229486358440136
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 69, valid acc = 67.220%, loss = 1.2020651547214654
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.420%, loss = 2.2474943579310347
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 70, valid acc = 67.170%, loss = 1.1788939418672006
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.472%, loss = 2.2620632822251383
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 71, valid acc = 66.120%, loss = 1.2927896229526665
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.434%, loss = 2.2567274521683793
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 72, valid acc = 66.680%, loss = 1.2216380095180077
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.040%, loss = 2.273032597873522
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 73, valid acc = 67.050%, loss = 1.1953092587145069
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.166%, loss = 2.2851120654274437
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 74, valid acc = 66.950%, loss = 1.1578480329694627
100%|██████████| 391/391 [02:23<00:00,  2.73it/s]
train acc = 67.070%, loss = 2.2806764722175306
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 75, valid acc = 66.090%, loss = 1.2392181256149388
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.934%, loss = 2.2875790650887255
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 76, valid acc = 65.870%, loss = 1.3002154110353203
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.300%, loss = 2.276498686322166
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 77, valid acc = 66.910%, loss = 1.1916506048999256
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.138%, loss = 2.2739858764516727
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 78, valid acc = 67.540%, loss = 1.1913161013699785
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.640%, loss = 2.2643368847839667
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
epoch = 79, valid acc = 67.400%, loss = 1.16920814106736
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.240%, loss = 2.2708945984730633
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 80, valid acc = 67.080%, loss = 1.1521919414966921
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.934%, loss = 2.226121278370128
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 81, valid acc = 68.380%, loss = 1.1936630380304554
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 67.958%, loss = 2.2199762471191717
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 82, valid acc = 67.930%, loss = 1.2493599902225445
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.152%, loss = 2.2083276799877587
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 83, valid acc = 67.980%, loss = 1.1701438517510137
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.914%, loss = 2.168247150033331
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 84, valid acc = 67.450%, loss = 1.2112043243420274
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 69.144%, loss = 2.152118679507614
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 85, valid acc = 67.310%, loss = 1.252823466741586
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.198%, loss = 2.1280331697000565
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 86, valid acc = 69.700%, loss = 1.1560634608510174
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 70.048%, loss = 2.0884347894917363
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 87, valid acc = 68.550%, loss = 1.1761756561979462
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.166%, loss = 2.068650863359651
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 88, valid acc = 68.250%, loss = 1.233665212045742
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.150%, loss = 2.0323905319813877
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 89, valid acc = 69.860%, loss = 1.1363122184065324
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 70.850%, loss = 2.021870272543729
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 90, valid acc = 69.960%, loss = 1.1056647134732596
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.262%, loss = 1.9958359161606225
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 91, valid acc = 70.030%, loss = 1.1187028481235988
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.880%, loss = 1.9600990678343322
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 92, valid acc = 69.700%, loss = 1.1656174976614457
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.200%, loss = 1.938546380423524
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 93, valid acc = 70.940%, loss = 1.1087114841123171
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.468%, loss = 1.9162600308732913
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 94, valid acc = 71.010%, loss = 1.0790577455411983
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.984%, loss = 1.9021854129288813
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 95, valid acc = 71.080%, loss = 1.1095647638357138
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.472%, loss = 1.862249588112697
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 96, valid acc = 70.830%, loss = 1.1382258383533623
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.312%, loss = 1.8616321580794157
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 97, valid acc = 71.170%, loss = 1.1155842768995068
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.498%, loss = 1.8463508186437894
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 98, valid acc = 71.400%, loss = 1.1035709373558624
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.874%, loss = 1.8386115411968182
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
epoch = 99, valid acc = 72.070%, loss = 1.0877569295183014
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.728%, loss = 1.8326803393985913
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 100, valid acc = 72.010%, loss = 1.1021067726461193
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.798%, loss = 1.826802861660033
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 101, valid acc = 71.500%, loss = 1.1051189393936833
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.766%, loss = 1.8276077807711824
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 102, valid acc = 71.850%, loss = 1.1120892838586736
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.464%, loss = 1.84647404507298
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 103, valid acc = 71.460%, loss = 1.1196133886711508
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.834%, loss = 1.842220660670639
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 104, valid acc = 71.530%, loss = 1.1179438693613946
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 73.262%, loss = 1.8577751079788598
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 105, valid acc = 72.140%, loss = 1.0698072050191179
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.912%, loss = 1.8864679617040299
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 106, valid acc = 71.000%, loss = 1.1418310592446146
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.824%, loss = 1.8834103614168094
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 107, valid acc = 71.200%, loss = 1.1202401805527602
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.678%, loss = 1.9089418323448553
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 108, valid acc = 70.910%, loss = 1.1327156126499176
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.474%, loss = 1.9114633204077212
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 109, valid acc = 70.580%, loss = 1.1194026251382465
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.402%, loss = 1.9310502896223531
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 110, valid acc = 70.060%, loss = 1.1425544581835783
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.104%, loss = 1.9467331296037835
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 111, valid acc = 70.410%, loss = 1.174855435573602
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.780%, loss = 1.9734484527422034
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 112, valid acc = 69.740%, loss = 1.1686837605283231
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.714%, loss = 1.9748823221992045
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 113, valid acc = 69.360%, loss = 1.2196024686475344
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.388%, loss = 1.9901997582686832
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 114, valid acc = 70.750%, loss = 1.1709179538714736
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.416%, loss = 1.9914834441431344
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 115, valid acc = 69.740%, loss = 1.188017258161231
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.336%, loss = 1.9921717741300382
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 116, valid acc = 71.840%, loss = 1.082116247732428
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.166%, loss = 2.0013069533326133
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 117, valid acc = 68.980%, loss = 1.2195154066327252
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.244%, loss = 2.000872919931436
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 118, valid acc = 70.390%, loss = 1.1288990778259085
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.668%, loss = 1.9840057710247576
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 119, valid acc = 70.150%, loss = 1.1126631415342982
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 71.404%, loss = 1.988590313040692
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 120, valid acc = 70.630%, loss = 1.0727375051643275
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.420%, loss = 1.973073509038257
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 121, valid acc = 71.090%, loss = 1.1078637612016895
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.000%, loss = 1.9578503056255447
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 122, valid acc = 70.750%, loss = 1.1498139055469367
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.914%, loss = 1.9456802512068883
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 123, valid acc = 71.250%, loss = 1.0627317768108995
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.224%, loss = 1.9280806377415767
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 124, valid acc = 69.960%, loss = 1.1843224477164354
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.790%, loss = 1.9042449713972829
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 125, valid acc = 69.950%, loss = 1.212624213363551
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 72.810%, loss = 1.889066010789798
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 126, valid acc = 72.100%, loss = 1.0374962046176572
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.206%, loss = 1.8690110441973753
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 127, valid acc = 72.510%, loss = 1.1028336667561833
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 73.714%, loss = 1.8323950618124374
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 128, valid acc = 72.080%, loss = 1.0872920012172265
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 73.886%, loss = 1.8168266700661702
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 129, valid acc = 71.590%, loss = 1.0866403753244425
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 74.316%, loss = 1.7894510652707971
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 130, valid acc = 71.790%, loss = 1.1745451786850072
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.920%, loss = 1.7541667948598447
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 131, valid acc = 72.060%, loss = 1.1337963802905022
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 75.322%, loss = 1.7308966918369693
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 132, valid acc = 73.240%, loss = 1.1134252080434486
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.620%, loss = 1.7010532962086866
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 133, valid acc = 72.590%, loss = 1.1448593018930169
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.906%, loss = 1.6788556834925776
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 134, valid acc = 73.090%, loss = 1.0776086638245401
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 76.300%, loss = 1.6586087951269906
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 135, valid acc = 73.660%, loss = 1.072151055441627
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.572%, loss = 1.6347230175876861
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 136, valid acc = 73.580%, loss = 1.087685050089148
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.918%, loss = 1.6244601166766623
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 137, valid acc = 73.890%, loss = 1.0804443291470975
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 77.034%, loss = 1.607067438342687
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 138, valid acc = 74.290%, loss = 1.06267577636091
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 77.040%, loss = 1.5987736444034235
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 139, valid acc = 74.250%, loss = 1.0827954343602628
100%|██████████| 391/391 [02:22<00:00,  2.75it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.188%, loss = 1.6032489809538701
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 140, valid acc = 73.980%, loss = 1.0867768028114415
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.800%, loss = 1.6115745659679404
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 141, valid acc = 73.740%, loss = 1.0790894982181019
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.834%, loss = 1.6111174303552378
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 142, valid acc = 73.580%, loss = 1.1034597588490835
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.952%, loss = 1.6136132734815787
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 143, valid acc = 74.100%, loss = 1.078850789160668
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 76.902%, loss = 1.622204486061545
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 144, valid acc = 73.870%, loss = 1.0881164835978159
100%|██████████| 391/391 [02:21<00:00,  2.75it/s]
train acc = 76.656%, loss = 1.635402273339079
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 145, valid acc = 74.240%, loss = 1.0722998395750793
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.572%, loss = 1.6390061424211468
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 146, valid acc = 73.730%, loss = 1.0766012313999707
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 76.338%, loss = 1.6638040365770346
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 147, valid acc = 74.100%, loss = 1.0566574289828916
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.914%, loss = 1.6795558950785177
100%|██████████| 79/79 [00:09<00:00,  8.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 148, valid acc = 73.300%, loss = 1.1105905122394804
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 75.814%, loss = 1.6828700508303045
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 149, valid acc = 72.950%, loss = 1.1553901547118077
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.324%, loss = 1.7201908314624406
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 150, valid acc = 73.410%, loss = 1.0723843819732908
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.466%, loss = 1.7303289065275655
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 151, valid acc = 74.040%, loss = 1.0177491133726095
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 74.972%, loss = 1.733787436619439
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 152, valid acc = 72.220%, loss = 1.1072850996934915
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.032%, loss = 1.7555682156092065
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 153, valid acc = 71.880%, loss = 1.1783014799975142
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.510%, loss = 1.787766825207664
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 154, valid acc = 72.410%, loss = 1.0701234834858133
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.516%, loss = 1.7821750564648366
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 155, valid acc = 72.360%, loss = 1.1265199056154565
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.278%, loss = 1.790330598421414
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 156, valid acc = 72.070%, loss = 1.0923651094678082
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.298%, loss = 1.7931709728582437
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 157, valid acc = 69.860%, loss = 1.2140058180953883
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.172%, loss = 1.7977907535670055
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
epoch = 158, valid acc = 72.230%, loss = 1.1237110865267017
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 74.236%, loss = 1.7999790574583556
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 159, valid acc = 72.960%, loss = 1.0720051904267902
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.020%, loss = 1.806705537964316
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 160, valid acc = 71.610%, loss = 1.1470896526228023
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.252%, loss = 1.7830540420454177
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 161, valid acc = 72.580%, loss = 1.084051428716394
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 74.410%, loss = 1.782261803022126
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 162, valid acc = 72.740%, loss = 1.083331680750545
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 74.778%, loss = 1.7589196692342344
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 163, valid acc = 73.320%, loss = 1.1567552670647827
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.018%, loss = 1.7389706504314453
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 164, valid acc = 73.390%, loss = 1.0361795629127115
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 75.136%, loss = 1.7322901600157208
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 165, valid acc = 73.080%, loss = 1.1095459876181204
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 75.614%, loss = 1.7089595992851745
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 166, valid acc = 74.160%, loss = 1.0258897165708905
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 76.014%, loss = 1.6800560030485967
100%|██████████| 79/79 [00:09<00:00,  8.59it/s]
epoch = 167, valid acc = 74.840%, loss = 1.0245074260083935
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.512%, loss = 1.6572537172176038
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 168, valid acc = 73.950%, loss = 1.0912019512321376
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
train acc = 76.748%, loss = 1.629918447235966
100%|██████████| 79/79 [00:09<00:00,  8.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 169, valid acc = 74.360%, loss = 1.0315425622312329
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.102%, loss = 1.6170212521272547
100%|██████████| 79/79 [00:09<00:00,  8.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 170, valid acc = 74.360%, loss = 1.1060223081443883
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.272%, loss = 1.5820989249002597
100%|██████████| 79/79 [00:09<00:00,  8.60it/s]
epoch = 171, valid acc = 74.660%, loss = 1.0733723074575015
100%|██████████| 391/391 [02:21<00:00,  2.76it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.772%, loss = 1.5536762695483235
100%|██████████| 79/79 [00:09<00:00,  8.58it/s]
epoch = 172, valid acc = 74.490%, loss = 1.0957868770708012
100%|██████████| 391/391 [02:47<00:00,  2.34it/s]
train acc = 78.342%, loss = 1.5206594773570594
100%|██████████| 79/79 [00:09<00:00,  8.13it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 173, valid acc = 74.950%, loss = 1.0529494013967393
  8%|▊         | 31/391 [00:11<02:16,  2.63it/s]
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.728%, loss = 1.4946692462467477
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 174, valid acc = 75.420%, loss = 1.0544521416289896
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 78.916%, loss = 1.47500003192126
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 175, valid acc = 75.640%, loss = 1.032804951637606
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.238%, loss = 1.4542858411588937
100%|██████████| 79/79 [00:09<00:00,  7.94it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 176, valid acc = 75.120%, loss = 1.0683774955665009
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.244%, loss = 1.448097972004005
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
epoch = 177, valid acc = 75.530%, loss = 1.067588435698159
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.364%, loss = 1.4436696020843427
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
epoch = 178, valid acc = 75.810%, loss = 1.02615650243397
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.562%, loss = 1.4336594629775532
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 179, valid acc = 75.480%, loss = 1.0653739201871655
100%|██████████| 391/391 [02:30<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.690%, loss = 1.427068237910795
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 180, valid acc = 75.570%, loss = 1.0442168248605124
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.854%, loss = 1.4209003689343973
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 181, valid acc = 75.200%, loss = 1.0879657728762566
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.848%, loss = 1.4239172584870283
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
epoch = 182, valid acc = 75.740%, loss = 1.0492005302936216
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.722%, loss = 1.431677817231249
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 183, valid acc = 75.510%, loss = 1.056772825084155
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 79.668%, loss = 1.4302047885897216
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 184, valid acc = 75.470%, loss = 1.039934602719319
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.478%, loss = 1.4526127709452148
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 185, valid acc = 75.420%, loss = 1.0635899607139299
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.184%, loss = 1.4604843618619778
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 186, valid acc = 75.000%, loss = 1.0829649944848652
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
train acc = 78.744%, loss = 1.4791156391963325
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 187, valid acc = 75.370%, loss = 1.074210611325276
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.608%, loss = 1.4937284163501867
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 188, valid acc = 75.590%, loss = 1.0492323704912692
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 78.370%, loss = 1.520198246707087
100%|██████████| 79/79 [00:09<00:00,  7.95it/s]
epoch = 189, valid acc = 74.470%, loss = 1.0622024279606492
100%|██████████| 391/391 [02:30<00:00,  2.61it/s]
train acc = 78.216%, loss = 1.5395494135444427
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 190, valid acc = 75.020%, loss = 1.0585188005543962
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.766%, loss = 1.5567534366227171
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 191, valid acc = 74.060%, loss = 1.1361707124529006
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 77.600%, loss = 1.570486117506881
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 192, valid acc = 73.990%, loss = 1.0847925607916675
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 77.212%, loss = 1.5964950606646136
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
epoch = 193, valid acc = 74.980%, loss = 1.054270555701437
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 76.894%, loss = 1.6121919633787307
100%|██████████| 79/79 [00:09<00:00,  7.91it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 194, valid acc = 74.230%, loss = 1.1342418352259864
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 76.654%, loss = 1.62293068557749
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 195, valid acc = 74.200%, loss = 1.1110121811492533
100%|██████████| 391/391 [02:30<00:00,  2.61it/s]
train acc = 76.598%, loss = 1.6279372519544324
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 196, valid acc = 74.540%, loss = 1.0576221128053303
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
train acc = 76.734%, loss = 1.6411374639672087
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 197, valid acc = 74.770%, loss = 1.1045984573002103
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.778%, loss = 1.6362332139173736
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
epoch = 198, valid acc = 74.280%, loss = 1.076531011092512
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 76.540%, loss = 1.6417875524676975
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 199, valid acc = 74.370%, loss = 1.0875352025032043
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.500%, loss = 1.644051151812229
100%|██████████| 79/79 [00:09<00:00,  8.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 200, valid acc = 72.810%, loss = 1.1253839308702493
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.684%, loss = 1.6326025334160652
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 201, valid acc = 74.430%, loss = 1.0970845577083057
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 76.762%, loss = 1.6308946164367755
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
epoch = 202, valid acc = 74.100%, loss = 1.1243127712720558
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.020%, loss = 1.6104172340134526
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 203, valid acc = 73.980%, loss = 1.0917922861968414
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.334%, loss = 1.5876385671708284
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 204, valid acc = 73.750%, loss = 1.1784355881847912
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.552%, loss = 1.5839927376383711
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 205, valid acc = 73.830%, loss = 1.1527660624890388
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 77.674%, loss = 1.5659871177600169
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
epoch = 206, valid acc = 74.490%, loss = 1.0780444265920905
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.238%, loss = 1.5415934610854634
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 207, valid acc = 75.050%, loss = 1.1085175841669492
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.294%, loss = 1.5206303120878957
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 208, valid acc = 75.500%, loss = 1.0144403086432927
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 78.906%, loss = 1.4826691822932505
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 209, valid acc = 75.210%, loss = 1.058498216580741
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.318%, loss = 1.4637145727796628
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
epoch = 210, valid acc = 75.460%, loss = 1.0797556699076785
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.946%, loss = 1.4305368035345736
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 211, valid acc = 75.070%, loss = 1.110316653040391
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.078%, loss = 1.401116574359367
100%|██████████| 79/79 [00:09<00:00,  7.93it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 212, valid acc = 76.680%, loss = 1.0335647546792333
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.432%, loss = 1.379273063234051
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 213, valid acc = 75.560%, loss = 1.0778775471675246
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.880%, loss = 1.3461286480469472
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 214, valid acc = 76.780%, loss = 1.0308830934234812
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.086%, loss = 1.3291102400826067
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 215, valid acc = 76.990%, loss = 1.0323508321484434
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.332%, loss = 1.3198285456508627
100%|██████████| 79/79 [00:09<00:00,  7.94it/s]
epoch = 216, valid acc = 75.920%, loss = 1.0854136649566362
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.548%, loss = 1.302432950195449
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 217, valid acc = 76.680%, loss = 1.0687577535834494
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.586%, loss = 1.2932377270115611
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 218, valid acc = 76.400%, loss = 1.073767723161963
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 81.992%, loss = 1.2837444383774876
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 219, valid acc = 77.220%, loss = 1.0443367611003826
100%|██████████| 391/391 [02:33<00:00,  2.55it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.080%, loss = 1.2798779134250358
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 220, valid acc = 76.630%, loss = 1.0441668516472926
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.940%, loss = 1.276847262821539
100%|██████████| 79/79 [00:09<00:00,  8.07it/s]
epoch = 221, valid acc = 76.460%, loss = 1.0729183139680307
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.774%, loss = 1.2800597923491008
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 222, valid acc = 76.560%, loss = 1.0418275655070437
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.898%, loss = 1.2840058539834474
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
epoch = 223, valid acc = 76.150%, loss = 1.073541444313677
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.654%, loss = 1.2991080268874498
100%|██████████| 79/79 [00:09<00:00,  8.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 224, valid acc = 76.450%, loss = 1.079304367681093
100%|██████████| 391/391 [02:28<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.426%, loss = 1.3132882881957246
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 225, valid acc = 76.300%, loss = 1.089447852931445
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.264%, loss = 1.3181641574405953
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 226, valid acc = 75.610%, loss = 1.1037950613830663
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.148%, loss = 1.336601197719574
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 227, valid acc = 75.950%, loss = 1.0615473751780353
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.664%, loss = 1.3636301074491437
100%|██████████| 79/79 [00:09<00:00,  8.06it/s]
epoch = 228, valid acc = 76.170%, loss = 1.0470175569570517
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.670%, loss = 1.3717453147444274
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 229, valid acc = 76.160%, loss = 1.0249766526342947
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.162%, loss = 1.3961041909654428
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 230, valid acc = 75.290%, loss = 1.0978267072122307
100%|██████████| 391/391 [02:28<00:00,  2.62it/s]
train acc = 79.798%, loss = 1.4097806616207522
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 231, valid acc = 74.930%, loss = 1.138145080095605
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.706%, loss = 1.4301823121507455
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
epoch = 232, valid acc = 75.470%, loss = 1.0857660740236692
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 79.210%, loss = 1.4645673152430894
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 233, valid acc = 73.410%, loss = 1.1981281110003024
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.814%, loss = 1.4785824447031826
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 234, valid acc = 74.410%, loss = 1.1625014898143238
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.952%, loss = 1.4945594121123214
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 235, valid acc = 74.860%, loss = 1.1076708881160882
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.652%, loss = 1.5072171755154113
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 236, valid acc = 75.100%, loss = 1.075959333890601
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.622%, loss = 1.506192340722779
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 237, valid acc = 74.330%, loss = 1.1256686184979692
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.346%, loss = 1.5204899210454252
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
epoch = 238, valid acc = 73.610%, loss = 1.1402199087263662
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 78.322%, loss = 1.5213962082972612
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 239, valid acc = 74.780%, loss = 1.1305221915245056
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 78.596%, loss = 1.516941048909941
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 240, valid acc = 74.600%, loss = 1.0975712990459008
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.082%, loss = 1.530874007498212
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 241, valid acc = 74.370%, loss = 1.1380340373968776
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 78.652%, loss = 1.5005772034530445
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 242, valid acc = 74.600%, loss = 1.073723602144024
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.916%, loss = 1.4901895437704022
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 243, valid acc = 74.510%, loss = 1.171945690731459
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 78.892%, loss = 1.4936764851555495
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 244, valid acc = 75.660%, loss = 1.0166766828374019
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 79.310%, loss = 1.4619986018561342
100%|██████████| 79/79 [00:09<00:00,  7.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 245, valid acc = 75.080%, loss = 1.1041412874113155
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
train acc = 79.418%, loss = 1.4425390246883987
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 246, valid acc = 74.760%, loss = 1.1318580938290945
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 79.918%, loss = 1.4106171469554267
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 247, valid acc = 75.940%, loss = 1.0809916974622993
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.426%, loss = 1.3793433486957989
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 248, valid acc = 74.990%, loss = 1.1318124547789368
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.696%, loss = 1.368322635550633
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
epoch = 249, valid acc = 74.920%, loss = 1.1341670630853387
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.010%, loss = 1.3462406651443228
100%|██████████| 79/79 [00:09<00:00,  8.06it/s]
epoch = 250, valid acc = 75.490%, loss = 1.106270384939411
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.528%, loss = 1.3230538999333101
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 251, valid acc = 76.550%, loss = 1.0665960734403586
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.798%, loss = 1.2836089777519635
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 252, valid acc = 77.600%, loss = 1.0336399757409398
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.334%, loss = 1.2620270610465418
100%|██████████| 79/79 [00:09<00:00,  7.90it/s]
epoch = 253, valid acc = 76.380%, loss = 1.0821738703341424
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.736%, loss = 1.2317427343419751
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 254, valid acc = 76.430%, loss = 1.1044991401177418
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.080%, loss = 1.2074904473846222
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 255, valid acc = 76.940%, loss = 1.0966291514378559
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.360%, loss = 1.1858864839729446
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 256, valid acc = 76.920%, loss = 1.085656295848798
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
train acc = 83.350%, loss = 1.1784584218888636
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 257, valid acc = 77.380%, loss = 1.0821342807781846
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.522%, loss = 1.1762152927001115
100%|██████████| 79/79 [00:09<00:00,  8.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 258, valid acc = 77.240%, loss = 1.0740641295155393
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.578%, loss = 1.1682779042007367
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 259, valid acc = 77.200%, loss = 1.0859727919856204
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.752%, loss = 1.162075048517388
100%|██████████| 79/79 [00:09<00:00,  8.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 260, valid acc = 77.480%, loss = 1.092579518692403
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.872%, loss = 1.1492029966600716
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 261, valid acc = 77.240%, loss = 1.0926153625868544
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
train acc = 83.906%, loss = 1.1618904950063857
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 262, valid acc = 76.930%, loss = 1.094790267793438
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.766%, loss = 1.1651730313325477
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 263, valid acc = 77.510%, loss = 1.1057373753076867
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.444%, loss = 1.1780182190258484
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 264, valid acc = 76.900%, loss = 1.1258868967430502
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.462%, loss = 1.1831302639773436
100%|██████████| 79/79 [00:09<00:00,  8.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 265, valid acc = 76.680%, loss = 1.1014571363412882
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.244%, loss = 1.2044353564377026
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 266, valid acc = 77.260%, loss = 1.0849770473528513
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.914%, loss = 1.2229598409989302
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 267, valid acc = 76.780%, loss = 1.0799748052524616
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.656%, loss = 1.2378938358153224
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
epoch = 268, valid acc = 76.370%, loss = 1.1586236357688904
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 82.416%, loss = 1.2563340232500335
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
epoch = 269, valid acc = 75.990%, loss = 1.111157532734207
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.068%, loss = 1.2768290417883403
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 270, valid acc = 76.370%, loss = 1.0749188563491725
100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.820%, loss = 1.2924025633450968
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
epoch = 271, valid acc = 75.370%, loss = 1.1639604606205904
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.384%, loss = 1.320540522370497
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 272, valid acc = 75.310%, loss = 1.1312085442905184
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.868%, loss = 1.3455183263629904
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 273, valid acc = 76.530%, loss = 1.0949111196059216
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.728%, loss = 1.3657694772991074
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
epoch = 274, valid acc = 76.540%, loss = 1.044787574795228
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.342%, loss = 1.386390116361096
100%|██████████| 79/79 [00:09<00:00,  7.91it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 275, valid acc = 76.280%, loss = 1.1119470754756202
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.150%, loss = 1.3990514897324544
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 276, valid acc = 76.170%, loss = 1.0096369287635707
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.044%, loss = 1.4106168437491902
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 277, valid acc = 75.860%, loss = 1.0413994426968731
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.060%, loss = 1.4081850704329704
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 278, valid acc = 75.740%, loss = 1.057756729518311
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.892%, loss = 1.4212650104861735
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 279, valid acc = 75.380%, loss = 1.1151071685024454
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 79.978%, loss = 1.4098031302852094
100%|██████████| 79/79 [00:09<00:00,  8.04it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 280, valid acc = 75.000%, loss = 1.1129554600655278
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.166%, loss = 1.4019576064156145
100%|██████████| 79/79 [00:09<00:00,  7.92it/s]
epoch = 281, valid acc = 75.590%, loss = 1.061981383003766
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.060%, loss = 1.3990032816177134
100%|██████████| 79/79 [00:09<00:00,  8.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 282, valid acc = 74.910%, loss = 1.1469032349465769
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.516%, loss = 1.3839137198979898
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 283, valid acc = 74.330%, loss = 1.1737416446963442
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.520%, loss = 1.3701039285915892
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 284, valid acc = 74.880%, loss = 1.1541471662400644
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 80.780%, loss = 1.3604498879074136
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 285, valid acc = 75.240%, loss = 1.1761347244057474
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.054%, loss = 1.3398279712328216
100%|██████████| 79/79 [00:09<00:00,  8.08it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 286, valid acc = 74.820%, loss = 1.1439705641963813
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.576%, loss = 1.316535420734864
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 287, valid acc = 75.640%, loss = 1.1031567295895348
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.720%, loss = 1.2936787487905654
100%|██████████| 79/79 [00:09<00:00,  7.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 288, valid acc = 76.740%, loss = 1.063252857968777
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 81.914%, loss = 1.2653911086299536
100%|██████████| 79/79 [00:09<00:00,  7.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 289, valid acc = 77.100%, loss = 1.0857059668891038
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
train acc = 82.566%, loss = 1.2402315008670777
100%|██████████| 79/79 [00:09<00:00,  7.94it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 290, valid acc = 76.190%, loss = 1.0590496900715405
100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
train acc = 83.216%, loss = 1.2046034352858659
100%|██████████| 79/79 [00:09<00:00,  7.98it/s]
epoch = 291, valid acc = 76.130%, loss = 1.1279600481443768
100%|██████████| 391/391 [02:51<00:00,  2.28it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.622%, loss = 1.1849737780173417
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 292, valid acc = 76.450%, loss = 1.1290008757687822
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.104%, loss = 1.1487213205498503
100%|██████████| 79/79 [00:09<00:00,  8.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 293, valid acc = 77.420%, loss = 1.0831310228456426
100%|██████████| 391/391 [02:30<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.018%, loss = 1.1428511795180534
100%|██████████| 79/79 [00:09<00:00,  7.95it/s]
epoch = 294, valid acc = 77.760%, loss = 1.0877741205541394
100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.698%, loss = 1.1044523505603565
100%|██████████| 79/79 [00:09<00:00,  8.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 295, valid acc = 77.340%, loss = 1.087338072589681
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.674%, loss = 1.0933444760644528
100%|██████████| 79/79 [00:09<00:00,  8.02it/s]
epoch = 296, valid acc = 78.090%, loss = 1.0695821152457707
100%|██████████| 391/391 [02:30<00:00,  2.61it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.172%, loss = 1.074315650993601
100%|██████████| 79/79 [00:09<00:00,  7.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 297, valid acc = 77.320%, loss = 1.1045046282719961
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.402%, loss = 1.0594020280081902
100%|██████████| 79/79 [00:09<00:00,  8.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 298, valid acc = 78.050%, loss = 1.087798782541782
100%|██████████| 391/391 [02:30<00:00,  2.60it/s]
train acc = 85.446%, loss = 1.0571119910310907
100%|██████████| 79/79 [00:09<00:00,  7.97it/s]
epoch = 299, valid acc = 78.080%, loss = 1.0742802925502197

```







```
100%|██████████| 391/391 [02:18<00:00,  2.83it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 21.212%, loss = 2.076728448233641
100%|██████████| 79/79 [01:38<00:00,  1.24s/it]
epoch = 0, valid acc = 11.010%, loss = 2.322188679176041
100%|██████████| 391/391 [02:20<00:00,  2.79it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 29.358%, loss = 1.8932933612247866
100%|██████████| 79/79 [01:37<00:00,  1.24s/it]
epoch = 1, valid acc = 9.770%, loss = 2.3308655068844177
100%|██████████| 391/391 [02:20<00:00,  2.78it/s]
train acc = 33.964%, loss = 1.7886553731415888
100%|██████████| 79/79 [01:37<00:00,  1.24s/it]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 2, valid acc = 12.580%, loss = 2.2552723341350314
100%|██████████| 391/391 [02:20<00:00,  2.78it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 36.536%, loss = 1.7250356244309175
100%|██████████| 79/79 [01:39<00:00,  1.26s/it]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 3, valid acc = 18.000%, loss = 2.126724698875524
100%|██████████| 391/391 [02:19<00:00,  2.80it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 39.036%, loss = 1.6681226153507867
100%|██████████| 79/79 [01:37<00:00,  1.24s/it]
epoch = 4, valid acc = 21.620%, loss = 2.0559103594550603
100%|██████████| 391/391 [02:19<00:00,  2.79it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 41.144%, loss = 1.617582588244582
100%|██████████| 79/79 [01:37<00:00,  1.24s/it]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 5, valid acc = 27.800%, loss = 1.9667926574055152
100%|██████████| 391/391 [02:19<00:00,  2.79it/s]
train acc = 42.894%, loss = 1.57112889216684
100%|██████████| 79/79 [01:38<00:00,  1.24s/it]
epoch = 6, valid acc = 33.730%, loss = 1.8154401779174805
100%|██████████| 391/391 [02:19<00:00,  2.80it/s]
train acc = 44.904%, loss = 1.5187017917633057
100%|██████████| 79/79 [01:37<00:00,  1.24s/it]
epoch = 7, valid acc = 39.280%, loss = 1.6808991522728642
100%|██████████| 391/391 [02:19<00:00,  2.79it/s]
train acc = 45.692%, loss = 1.489333716804719
100%|██████████| 79/79 [01:38<00:00,  1.24s/it]
epoch = 8, valid acc = 42.520%, loss = 1.5843474170829677
100%|██████████| 391/391 [02:20<00:00,  2.79it/s]
train acc = 46.984%, loss = 1.4549621837523283
100%|██████████| 79/79 [01:39<00:00,  1.25s/it]
epoch = 9, valid acc = 44.410%, loss = 1.5283805222450932


100%|██████████| 391/391 [01:13<00:00,  5.35it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 19.474%, loss = 2.1156251247581617
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
epoch = 0, valid acc = 10.650%, loss = 2.3290886667710318
100%|██████████| 391/391 [01:12<00:00,  5.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 27.392%, loss = 1.9460231849299672
100%|██████████| 79/79 [00:12<00:00,  6.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 1, valid acc = 11.010%, loss = 2.2814175992072383
100%|██████████| 391/391 [01:12<00:00,  5.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 31.430%, loss = 1.8434684118041602
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 2, valid acc = 18.160%, loss = 2.2326940041554124
100%|██████████| 391/391 [01:12<00:00,  5.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 34.526%, loss = 1.762410514494952
100%|██████████| 79/79 [00:11<00:00,  6.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 3, valid acc = 20.120%, loss = 2.181467804727675
100%|██████████| 391/391 [01:12<00:00,  5.37it/s]
train acc = 37.562%, loss = 1.6878612517091014
100%|██████████| 79/79 [00:11<00:00,  6.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 4, valid acc = 23.290%, loss = 2.0851972450183918
100%|██████████| 391/391 [01:12<00:00,  5.37it/s]
train acc = 41.004%, loss = 1.6186591920340458
100%|██████████| 79/79 [00:11<00:00,  6.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 5, valid acc = 27.290%, loss = 1.9689447246020353
100%|██████████| 391/391 [01:12<00:00,  5.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 42.980%, loss = 1.5625337804369914
100%|██████████| 79/79 [00:11<00:00,  6.63it/s]
epoch = 6, valid acc = 32.790%, loss = 1.834109828441958
100%|██████████| 391/391 [01:12<00:00,  5.37it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 45.068%, loss = 1.514450867462646
100%|██████████| 79/79 [00:12<00:00,  6.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 7, valid acc = 38.120%, loss = 1.7096512347837038
100%|██████████| 391/391 [01:20<00:00,  4.88it/s]
train acc = 46.352%, loss = 1.4753862628546517
100%|██████████| 79/79 [00:13<00:00,  6.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 8, valid acc = 42.350%, loss = 1.5906864359408994
100%|██████████| 391/391 [01:19<00:00,  4.90it/s]
train acc = 47.400%, loss = 1.4430274493858943
100%|██████████| 79/79 [00:13<00:00,  6.01it/s]
epoch = 9, valid acc = 45.290%, loss = 1.5086144827589203
100%|██████████| 391/391 [01:13<00:00,  5.34it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 48.776%, loss = 1.4120610388343597
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 10, valid acc = 47.230%, loss = 1.4604754025423075
100%|██████████| 391/391 [01:20<00:00,  4.84it/s]
train acc = 50.072%, loss = 1.381271318401522
100%|██████████| 79/79 [00:13<00:00,  6.02it/s]
epoch = 11, valid acc = 48.880%, loss = 1.4107209293148186
100%|██████████| 391/391 [01:19<00:00,  4.91it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 50.866%, loss = 1.356920110600074
100%|██████████| 79/79 [00:13<00:00,  6.00it/s]
epoch = 12, valid acc = 50.140%, loss = 1.3762514727025092
100%|██████████| 391/391 [01:19<00:00,  4.90it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 51.506%, loss = 1.336264299614655
100%|██████████| 79/79 [00:12<00:00,  6.35it/s]
epoch = 13, valid acc = 51.200%, loss = 1.3518804025046434
100%|██████████| 391/391 [01:24<00:00,  4.64it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 51.994%, loss = 1.3212391436862214
100%|██████████| 79/79 [00:14<00:00,  5.55it/s]
epoch = 14, valid acc = 52.000%, loss = 1.335212853890431
100%|██████████| 391/391 [01:26<00:00,  4.51it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.814%, loss = 1.3059552490253887
100%|██████████| 79/79 [00:14<00:00,  5.55it/s]
epoch = 15, valid acc = 52.160%, loss = 1.3276012441779994
100%|██████████| 391/391 [01:26<00:00,  4.52it/s]
train acc = 53.194%, loss = 1.2917065666154828
100%|██████████| 79/79 [00:14<00:00,  5.52it/s]
epoch = 16, valid acc = 53.020%, loss = 1.3128460343879989
100%|██████████| 391/391 [01:27<00:00,  4.46it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.348%, loss = 1.2865914511863532
100%|██████████| 79/79 [00:14<00:00,  5.38it/s]
epoch = 17, valid acc = 52.820%, loss = 1.3121372959281825
100%|██████████| 391/391 [02:17<00:00,  2.83it/s]
train acc = 53.654%, loss = 1.2787351022900828
100%|██████████| 79/79 [00:26<00:00,  2.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 18, valid acc = 53.530%, loss = 1.2984118884122824
100%|██████████| 391/391 [02:37<00:00,  2.48it/s]
train acc = 53.884%, loss = 1.2754172431233595
100%|██████████| 79/79 [00:26<00:00,  2.98it/s]
epoch = 19, valid acc = 53.910%, loss = 1.2921055404445794
100%|██████████| 391/391 [01:50<00:00,  3.54it/s]
train acc = 53.978%, loss = 1.2727759494196118
100%|██████████| 79/79 [00:16<00:00,  4.72it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 20, valid acc = 53.040%, loss = 1.29736264144318
100%|██████████| 391/391 [02:19<00:00,  2.81it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.224%, loss = 1.2745948639672127
100%|██████████| 79/79 [00:22<00:00,  3.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 21, valid acc = 53.420%, loss = 1.291593996784355
100%|██████████| 391/391 [02:16<00:00,  2.86it/s]
train acc = 53.866%, loss = 1.2765698606705727
100%|██████████| 79/79 [00:22<00:00,  3.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 22, valid acc = 53.570%, loss = 1.2942576287667962
100%|██████████| 391/391 [02:10<00:00,  3.00it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.718%, loss = 1.2773310507045073
100%|██████████| 79/79 [00:22<00:00,  3.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 23, valid acc = 53.390%, loss = 1.291979033735734
100%|██████████| 391/391 [02:13<00:00,  2.94it/s]
train acc = 53.652%, loss = 1.280396929482365
100%|██████████| 79/79 [00:20<00:00,  3.92it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 24, valid acc = 53.890%, loss = 1.288141127628616
100%|██████████| 391/391 [01:32<00:00,  4.21it/s]
train acc = 53.672%, loss = 1.2796545394546235
100%|██████████| 79/79 [00:17<00:00,  4.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 25, valid acc = 53.910%, loss = 1.2832829363738434
100%|██████████| 391/391 [01:52<00:00,  3.48it/s]
train acc = 53.814%, loss = 1.2767007896662368
100%|██████████| 79/79 [00:18<00:00,  4.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 26, valid acc = 53.560%, loss = 1.287334351599971
100%|██████████| 391/391 [01:56<00:00,  3.37it/s]
train acc = 53.574%, loss = 1.2801686003994759
100%|██████████| 79/79 [00:18<00:00,  4.20it/s]
epoch = 27, valid acc = 54.200%, loss = 1.2804635431193099
100%|██████████| 391/391 [01:52<00:00,  3.46it/s]
train acc = 53.882%, loss = 1.2758215284713394
100%|██████████| 79/79 [00:20<00:00,  3.79it/s]
epoch = 28, valid acc = 54.010%, loss = 1.2745635758472393
100%|██████████| 391/391 [01:59<00:00,  3.28it/s]
train acc = 54.198%, loss = 1.2705703668887047
100%|██████████| 79/79 [00:18<00:00,  4.34it/s]
epoch = 29, valid acc = 54.450%, loss = 1.2600533388837982
100%|██████████| 391/391 [01:57<00:00,  3.32it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.406%, loss = 1.271434405880511
100%|██████████| 79/79 [00:19<00:00,  3.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 30, valid acc = 54.530%, loss = 1.2579109306576886
100%|██████████| 391/391 [01:50<00:00,  3.54it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.606%, loss = 1.2622696499690376
100%|██████████| 79/79 [00:20<00:00,  3.95it/s]
epoch = 31, valid acc = 55.030%, loss = 1.2464422877830794
100%|██████████| 391/391 [01:45<00:00,  3.69it/s]
train acc = 54.560%, loss = 1.2588891794004708
100%|██████████| 79/79 [00:18<00:00,  4.30it/s]
epoch = 32, valid acc = 55.600%, loss = 1.2358408321308185
100%|██████████| 391/391 [01:53<00:00,  3.44it/s]
train acc = 55.222%, loss = 1.2529739845744179
100%|██████████| 79/79 [00:17<00:00,  4.55it/s]
epoch = 33, valid acc = 56.260%, loss = 1.2298560972455181
100%|██████████| 391/391 [01:54<00:00,  3.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 55.280%, loss = 1.2392965167989511
100%|██████████| 79/79 [00:16<00:00,  4.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 34, valid acc = 56.040%, loss = 1.2216694943512543
100%|██████████| 391/391 [01:50<00:00,  3.55it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 55.664%, loss = 1.228422031835522
100%|██████████| 79/79 [00:19<00:00,  4.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 35, valid acc = 56.820%, loss = 1.2053753016870232
100%|██████████| 391/391 [01:51<00:00,  3.49it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.110%, loss = 1.2180386812180815
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
epoch = 36, valid acc = 57.550%, loss = 1.1853177389012108
100%|██████████| 391/391 [01:51<00:00,  3.50it/s]
train acc = 56.794%, loss = 1.209051626417643
100%|██████████| 79/79 [00:18<00:00,  4.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 37, valid acc = 57.360%, loss = 1.1769954743264597
100%|██████████| 391/391 [01:56<00:00,  3.36it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 57.130%, loss = 1.1942920759510811
100%|██████████| 79/79 [00:20<00:00,  3.81it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 38, valid acc = 58.630%, loss = 1.1591813481306728
100%|██████████| 391/391 [01:51<00:00,  3.50it/s]
train acc = 57.466%, loss = 1.1798297154629016
100%|██████████| 79/79 [00:16<00:00,  4.91it/s]
epoch = 39, valid acc = 58.750%, loss = 1.152766395973254
100%|██████████| 391/391 [01:53<00:00,  3.43it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 58.016%, loss = 1.1646744707966095
100%|██████████| 79/79 [00:19<00:00,  3.97it/s]
epoch = 40, valid acc = 59.350%, loss = 1.1377169150340407
100%|██████████| 391/391 [01:49<00:00,  3.58it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 58.946%, loss = 1.1501372650151362
100%|██████████| 79/79 [00:17<00:00,  4.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 41, valid acc = 59.880%, loss = 1.1238973185985903
100%|██████████| 391/391 [01:45<00:00,  3.69it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.714%, loss = 1.1257991411192032
100%|██████████| 79/79 [00:13<00:00,  5.65it/s]
epoch = 42, valid acc = 60.050%, loss = 1.1131299583217766
100%|██████████| 391/391 [01:25<00:00,  4.59it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 60.246%, loss = 1.1130322001474289
100%|██████████| 79/79 [00:12<00:00,  6.49it/s]
epoch = 43, valid acc = 60.410%, loss = 1.1051243536079987

```

































```
100%|██████████| 391/391 [04:12<00:00,  1.55it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 18.910%, loss = 5.144526949928849
100%|██████████| 79/79 [00:28<00:00,  2.76it/s]
epoch = 0, valid acc = 10.110%, loss = 2.307036016560808
100%|██████████| 391/391 [04:51<00:00,  1.34it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 27.050%, loss = 4.725642329896503
100%|██████████| 79/79 [00:19<00:00,  3.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 1, valid acc = 17.500%, loss = 2.197799438162695
100%|██████████| 391/391 [04:45<00:00,  1.37it/s]
train acc = 30.944%, loss = 4.502180475103276
100%|██████████| 79/79 [00:12<00:00,  6.46it/s]
epoch = 2, valid acc = 19.910%, loss = 2.1679373179809955
100%|██████████| 391/391 [04:33<00:00,  1.43it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 34.946%, loss = 4.272833752205305
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 3, valid acc = 24.870%, loss = 2.0327524203288405
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 38.032%, loss = 4.085721178127981
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 4, valid acc = 28.860%, loss = 1.9352188593224635
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 40.894%, loss = 3.956969219705333
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
epoch = 5, valid acc = 33.890%, loss = 1.8395507290393491
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 42.998%, loss = 3.84333382969927
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 6, valid acc = 37.770%, loss = 1.736171850675269
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 44.854%, loss = 3.7403364602257225
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 7, valid acc = 41.200%, loss = 1.6633123717730558
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
train acc = 45.730%, loss = 3.6612201249203107
100%|██████████| 79/79 [00:19<00:00,  4.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 8, valid acc = 43.330%, loss = 1.6180664783791652
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 47.182%, loss = 3.5716596722907727
100%|██████████| 79/79 [00:19<00:00,  4.09it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 9, valid acc = 45.110%, loss = 1.6132311760624753
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 48.122%, loss = 3.5060719718103823
100%|██████████| 79/79 [00:20<00:00,  3.85it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 10, valid acc = 46.600%, loss = 1.6008744752859767
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 49.002%, loss = 3.4446639746351315
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 11, valid acc = 47.930%, loss = 1.5866702991195871
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 50.228%, loss = 3.3881916481515635
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 12, valid acc = 48.410%, loss = 1.5854850117164323
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 50.632%, loss = 3.3485121623329492
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 13, valid acc = 49.240%, loss = 1.5657461881637573
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 51.144%, loss = 3.306134308085722
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 14, valid acc = 49.790%, loss = 1.5756585266016707
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.032%, loss = 3.271124881856582
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 15, valid acc = 50.450%, loss = 1.5714924999430209
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 52.310%, loss = 3.2539628119115025
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 16, valid acc = 50.920%, loss = 1.570116400718689
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 52.768%, loss = 3.226551894946476
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 17, valid acc = 51.140%, loss = 1.5722630582278287
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.040%, loss = 3.2109179393104883
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
epoch = 18, valid acc = 51.210%, loss = 1.585217474382135
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 53.232%, loss = 3.200139693896789
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 19, valid acc = 51.390%, loss = 1.571563651290121
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.886%, loss = 3.2058668313428873
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 20, valid acc = 51.970%, loss = 1.5575107064428209
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.276%, loss = 3.1912919523770853
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 21, valid acc = 51.550%, loss = 1.5744151299512839
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.000%, loss = 3.208016165077229
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 22, valid acc = 51.800%, loss = 1.5644849267186998
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.918%, loss = 3.2098919589196324
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 23, valid acc = 51.850%, loss = 1.5857291010361683
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.046%, loss = 3.197871645393274
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 24, valid acc = 51.710%, loss = 1.5504608425912978
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 52.980%, loss = 3.202426759178376
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 25, valid acc = 52.640%, loss = 1.5692393628856804
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.098%, loss = 3.2103532781381436
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 26, valid acc = 51.950%, loss = 1.5571136142634139
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 53.052%, loss = 3.1926731490113243
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 27, valid acc = 52.670%, loss = 1.5381704357605945
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 53.074%, loss = 3.2031125046712967
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 28, valid acc = 52.790%, loss = 1.537485905840427
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 53.326%, loss = 3.190504773498496
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 29, valid acc = 52.800%, loss = 1.5328352466414246
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 53.210%, loss = 3.168383111124453
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 30, valid acc = 53.240%, loss = 1.5280637876896919
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 53.270%, loss = 3.1708358096344695
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 31, valid acc = 53.590%, loss = 1.5038643909406058
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 53.972%, loss = 3.143856033949596
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 32, valid acc = 54.060%, loss = 1.497185624098476
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.632%, loss = 3.108950588099487
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 33, valid acc = 54.760%, loss = 1.4933228613455085
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 54.760%, loss = 3.0937011345573096
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 34, valid acc = 54.760%, loss = 1.4668220115613333
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 55.044%, loss = 3.070325804488433
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 35, valid acc = 55.140%, loss = 1.4588622564001927
100%|██████████| 391/391 [04:44<00:00,  1.37it/s]
train acc = 56.044%, loss = 3.034349756777439
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 36, valid acc = 55.610%, loss = 1.4540225946450536
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 56.034%, loss = 3.009546802476849
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 37, valid acc = 56.250%, loss = 1.4493927668921556
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 56.770%, loss = 2.96641308450333
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 38, valid acc = 56.950%, loss = 1.425052765049512
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 57.324%, loss = 2.9395822246971033
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 39, valid acc = 57.540%, loss = 1.4093948195252237
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 57.806%, loss = 2.9003852291790118
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 40, valid acc = 58.230%, loss = 1.4000635388531262
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 58.270%, loss = 2.85756719386791
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 41, valid acc = 58.730%, loss = 1.3923758358895024
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.270%, loss = 2.811587884298066
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 42, valid acc = 59.150%, loss = 1.388189699076399
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 59.374%, loss = 2.7933056787456696
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 43, valid acc = 59.620%, loss = 1.3687178847155994
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 60.156%, loss = 2.7374869246616997
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 44, valid acc = 59.510%, loss = 1.3750714561607265
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 61.030%, loss = 2.6938470178248024
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 45, valid acc = 59.900%, loss = 1.3721733078171936
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 61.736%, loss = 2.6444808150191443
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 46, valid acc = 60.670%, loss = 1.3653391433667532
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 62.044%, loss = 2.6173158179768516
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 47, valid acc = 60.960%, loss = 1.3446317714980887
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 62.928%, loss = 2.5795311689986598
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
epoch = 48, valid acc = 61.520%, loss = 1.340308302565466
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 63.448%, loss = 2.5389778925024946
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
epoch = 49, valid acc = 61.790%, loss = 1.3388163145584395
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
train acc = 64.118%, loss = 2.502984180779713
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 50, valid acc = 62.910%, loss = 1.3266023985947235
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 64.850%, loss = 2.4527672334095403
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 51, valid acc = 62.450%, loss = 1.3302982660788525
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.366%, loss = 2.4212985276566137
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 52, valid acc = 62.550%, loss = 1.314453531669665
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 65.770%, loss = 2.382435820291719
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 53, valid acc = 63.080%, loss = 1.3116469549227365
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
train acc = 66.400%, loss = 2.341268673577272
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 54, valid acc = 63.110%, loss = 1.3284127591531487
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.706%, loss = 2.3178439835453277
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 55, valid acc = 63.350%, loss = 1.313822250577468
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.066%, loss = 2.291652430963638
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 56, valid acc = 63.660%, loss = 1.3305919011936913
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 67.376%, loss = 2.2733906239194943
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 57, valid acc = 63.640%, loss = 1.3216660656506503
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 67.542%, loss = 2.2596471638935607
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 58, valid acc = 63.300%, loss = 1.3296397909333435
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 67.562%, loss = 2.2454062529537073
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 59, valid acc = 63.700%, loss = 1.3271144696428805
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.784%, loss = 2.238138364708942
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 60, valid acc = 63.560%, loss = 1.338924501515642
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.840%, loss = 2.2428098267606456
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 61, valid acc = 63.650%, loss = 1.3376883770091623
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.754%, loss = 2.2397856608681055
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 62, valid acc = 63.870%, loss = 1.3294640206083466
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.552%, loss = 2.250296542711575
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 63, valid acc = 64.120%, loss = 1.3385931381696388
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.456%, loss = 2.265730113934373
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 64, valid acc = 63.840%, loss = 1.3435394462150863
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.326%, loss = 2.284567058848603
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
epoch = 65, valid acc = 63.920%, loss = 1.3296392582639862
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 67.148%, loss = 2.289011679342031
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 66, valid acc = 64.290%, loss = 1.3309725138205517
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.970%, loss = 2.3013323359477247
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 67, valid acc = 64.340%, loss = 1.3342264437977271
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.876%, loss = 2.3160227421299577
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 68, valid acc = 64.490%, loss = 1.3166173643703702
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 66.812%, loss = 2.31673192459604
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 69, valid acc = 64.190%, loss = 1.323078120811076
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.630%, loss = 2.314090827237005
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 70, valid acc = 64.170%, loss = 1.3262078339540506
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.310%, loss = 2.3341942077402567
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 71, valid acc = 63.880%, loss = 1.2991917970814282
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 66.420%, loss = 2.3434103337090337
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
epoch = 72, valid acc = 64.420%, loss = 1.2858804471885101
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.292%, loss = 2.3417840821053977
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 73, valid acc = 64.620%, loss = 1.296829633320434
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.428%, loss = 2.343117564230624
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 74, valid acc = 65.330%, loss = 1.2774642328672772
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.576%, loss = 2.334719392649658
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 75, valid acc = 65.050%, loss = 1.2636680965182148
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.258%, loss = 2.336617069780979
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
epoch = 76, valid acc = 65.490%, loss = 1.2603412180007258
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 66.796%, loss = 2.309971794143052
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 77, valid acc = 65.610%, loss = 1.258242515823509
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 66.928%, loss = 2.302933230729359
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 78, valid acc = 66.060%, loss = 1.2357307257531565
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 67.446%, loss = 2.2738807231873808
100%|██████████| 79/79 [00:19<00:00,  4.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 79, valid acc = 65.800%, loss = 1.252818330179287
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 67.756%, loss = 2.242281632350229
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 80, valid acc = 66.220%, loss = 1.2281598097915891
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 68.062%, loss = 2.227153739965785
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 81, valid acc = 66.690%, loss = 1.2268425656270376
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.246%, loss = 2.199281121763732
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 82, valid acc = 67.100%, loss = 1.2220552729654917
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 68.646%, loss = 2.1854491139311927
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 83, valid acc = 67.170%, loss = 1.2253700019438056
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.312%, loss = 2.1445449752271024
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 84, valid acc = 67.320%, loss = 1.2267615478250045
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 69.726%, loss = 2.1053637543602672
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 85, valid acc = 68.130%, loss = 1.207279596902147
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
train acc = 70.198%, loss = 2.070807245686231
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 86, valid acc = 68.340%, loss = 1.2224492491046084
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 70.814%, loss = 2.0394003205287183
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 87, valid acc = 68.460%, loss = 1.200341514394253
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 71.690%, loss = 1.9871993860625246
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 88, valid acc = 68.480%, loss = 1.2150966336455526
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 72.064%, loss = 1.9578460985437378
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 89, valid acc = 68.500%, loss = 1.2209218159506592
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 72.656%, loss = 1.904436260538028
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 90, valid acc = 69.360%, loss = 1.1871596755860727
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 73.282%, loss = 1.860339462604669
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 91, valid acc = 69.020%, loss = 1.2136138375801375
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.082%, loss = 1.8269728214844414
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 92, valid acc = 69.250%, loss = 1.2250508243524576
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.616%, loss = 1.7777287289309684
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
epoch = 93, valid acc = 69.770%, loss = 1.1964584558824949
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.090%, loss = 1.7581588188400659
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 94, valid acc = 70.120%, loss = 1.214332969128331
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 75.790%, loss = 1.714047755731646
100%|██████████| 79/79 [00:18<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 95, valid acc = 70.080%, loss = 1.2282834581181974
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.900%, loss = 1.693248866769054
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 96, valid acc = 69.780%, loss = 1.2447998252096055
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.546%, loss = 1.6669458769776326
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 97, valid acc = 69.790%, loss = 1.2452347240870512
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 76.256%, loss = 1.6656478527561782
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 98, valid acc = 69.470%, loss = 1.2607689333867422
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 76.522%, loss = 1.652983347168359
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 99, valid acc = 69.880%, loss = 1.240955927703954
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
train acc = 76.622%, loss = 1.6459349863364567
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 100, valid acc = 70.380%, loss = 1.2434838881975487
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.664%, loss = 1.6542762249631955
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 101, valid acc = 70.390%, loss = 1.2411353301398362
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 76.510%, loss = 1.6544225664090013
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
epoch = 102, valid acc = 70.360%, loss = 1.25318891564502
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 76.398%, loss = 1.6583481892905272
100%|██████████| 79/79 [00:19<00:00,  4.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 103, valid acc = 70.420%, loss = 1.2335878965220874
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.240%, loss = 1.674245991365379
100%|██████████| 79/79 [00:19<00:00,  4.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 104, valid acc = 70.730%, loss = 1.2372684614567817
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 76.060%, loss = 1.683359628443218
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 105, valid acc = 70.860%, loss = 1.2453746539127977
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.910%, loss = 1.6945459931097981
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 106, valid acc = 70.400%, loss = 1.2396324903150149
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.482%, loss = 1.7247493169496737
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 107, valid acc = 70.060%, loss = 1.2504287807247307
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 75.464%, loss = 1.7263506899404404
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 108, valid acc = 70.610%, loss = 1.2516573079024689
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 75.104%, loss = 1.7379924342455462
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 109, valid acc = 70.390%, loss = 1.2404645470124256
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 74.674%, loss = 1.7614702492418801
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 110, valid acc = 70.320%, loss = 1.2185409423671192
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.416%, loss = 1.785132825831928
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 111, valid acc = 70.580%, loss = 1.2156283900707583
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.220%, loss = 1.7968064967323751
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 112, valid acc = 70.430%, loss = 1.2179191957546185
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.138%, loss = 1.8039880554999232
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 113, valid acc = 70.790%, loss = 1.2205629258216182
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 74.396%, loss = 1.8048410019301393
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 114, valid acc = 70.440%, loss = 1.2119324795807465
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.086%, loss = 1.8183444933513242
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 115, valid acc = 71.030%, loss = 1.2084590845470187
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 73.996%, loss = 1.8244677608275353
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 116, valid acc = 71.380%, loss = 1.1977436542510986
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 73.950%, loss = 1.8227346693463338
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 117, valid acc = 71.250%, loss = 1.1801436641548253
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 73.936%, loss = 1.8191996712209013
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 118, valid acc = 72.060%, loss = 1.1675255132626883
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 74.188%, loss = 1.8216870632927742
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
epoch = 119, valid acc = 72.040%, loss = 1.1479466632951665
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
train acc = 74.452%, loss = 1.788569368669749
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 120, valid acc = 71.670%, loss = 1.1760736100281342
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 74.294%, loss = 1.7937936203559037
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 121, valid acc = 71.770%, loss = 1.1436449021478243
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 74.744%, loss = 1.7658440706979892
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 122, valid acc = 71.580%, loss = 1.1622191954262648
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.150%, loss = 1.745421137346331
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 123, valid acc = 72.410%, loss = 1.1515975911406022
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 75.448%, loss = 1.7256501900875354
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 124, valid acc = 72.130%, loss = 1.1408198257035846
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 76.020%, loss = 1.689600077126642
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 125, valid acc = 72.000%, loss = 1.1413357544548903
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.422%, loss = 1.6596428838837178
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 126, valid acc = 71.750%, loss = 1.1597009874597382
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 76.646%, loss = 1.6333144038839413
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 127, valid acc = 72.420%, loss = 1.1733256729343269
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 77.608%, loss = 1.5812223044502767
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 128, valid acc = 72.620%, loss = 1.1622745304168025
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 77.942%, loss = 1.5489995888127086
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 129, valid acc = 72.640%, loss = 1.1558899615384355
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 78.448%, loss = 1.5095740994224158
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 130, valid acc = 72.270%, loss = 1.1905194173885296
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 79.100%, loss = 1.4670931299019347
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 131, valid acc = 73.030%, loss = 1.19220964968959
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 79.618%, loss = 1.4322895640912263
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 132, valid acc = 73.210%, loss = 1.1932367961617965
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 80.190%, loss = 1.396501319640128
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 133, valid acc = 73.110%, loss = 1.1999231629733798
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.776%, loss = 1.36256262408498
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 134, valid acc = 73.280%, loss = 1.186924588076676
100%|██████████| 391/391 [04:41<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.400%, loss = 1.3223505661920514
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 135, valid acc = 73.260%, loss = 1.2151655833932418
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 81.718%, loss = 1.306396760446641
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 136, valid acc = 73.490%, loss = 1.216193312331091
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 82.076%, loss = 1.2821650192560747
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 137, valid acc = 73.510%, loss = 1.2098414482949655
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.368%, loss = 1.2611759594639245
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 138, valid acc = 73.260%, loss = 1.2383477906637554
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
train acc = 82.470%, loss = 1.2552386581745294
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 139, valid acc = 73.230%, loss = 1.2394644278514235
100%|██████████| 391/391 [04:42<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.516%, loss = 1.2475828779932787
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 140, valid acc = 73.460%, loss = 1.231328784664975
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 82.802%, loss = 1.2404600909298948
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
epoch = 141, valid acc = 73.320%, loss = 1.247463337982757
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.270%, loss = 1.2632259368286718
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 142, valid acc = 73.110%, loss = 1.2795030165322219
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.248%, loss = 1.2649797303292452
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 143, valid acc = 73.120%, loss = 1.260716117635558
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.096%, loss = 1.2743200836584085
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 144, valid acc = 73.380%, loss = 1.2706262884260733
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.716%, loss = 1.2835900024379916
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 145, valid acc = 73.490%, loss = 1.2381889993631388
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.576%, loss = 1.3074727901412397
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
epoch = 146, valid acc = 73.820%, loss = 1.245993006832992
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.062%, loss = 1.3295103029521835
100%|██████████| 79/79 [00:19<00:00,  4.13it/s]
epoch = 147, valid acc = 73.560%, loss = 1.2400100314164464
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 80.926%, loss = 1.3536852433553437
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 148, valid acc = 73.440%, loss = 1.2206691790230666
100%|██████████| 391/391 [04:37<00:00,  1.41it/s]
train acc = 80.594%, loss = 1.3790394668383976
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 149, valid acc = 73.530%, loss = 1.2337056534199775
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
train acc = 80.362%, loss = 1.3951039628299606
100%|██████████| 79/79 [00:19<00:00,  4.13it/s]
epoch = 150, valid acc = 73.250%, loss = 1.2531712975683091
100%|██████████| 391/391 [04:38<00:00,  1.41it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.854%, loss = 1.4289087535780105
100%|██████████| 79/79 [00:19<00:00,  4.13it/s]
epoch = 151, valid acc = 73.620%, loss = 1.209973255290261
100%|██████████| 391/391 [04:38<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.396%, loss = 1.4477416889746781
100%|██████████| 79/79 [00:19<00:00,  4.13it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 152, valid acc = 74.450%, loss = 1.1820879246615157
100%|██████████| 391/391 [04:39<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.292%, loss = 1.4623274531815669
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 153, valid acc = 73.710%, loss = 1.1935224533081055
100%|██████████| 391/391 [04:45<00:00,  1.37it/s]
train acc = 79.078%, loss = 1.4881916366269827
100%|██████████| 79/79 [00:19<00:00,  4.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 154, valid acc = 73.710%, loss = 1.1775688181949566
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.722%, loss = 1.4938990307585966
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 155, valid acc = 73.860%, loss = 1.1736894189556943
100%|██████████| 391/391 [04:40<00:00,  1.39it/s]
train acc = 79.030%, loss = 1.4940681520020564
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
epoch = 156, valid acc = 74.120%, loss = 1.13786374164533
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 78.610%, loss = 1.5162959580531206
100%|██████████| 79/79 [00:19<00:00,  4.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 157, valid acc = 74.090%, loss = 1.1465992859647245
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.668%, loss = 1.4997947397439375
100%|██████████| 79/79 [00:19<00:00,  4.11it/s]
epoch = 158, valid acc = 74.010%, loss = 1.1457718663577792
100%|██████████| 391/391 [04:40<00:00,  1.40it/s]
train acc = 78.598%, loss = 1.5089017055223666
100%|██████████| 79/79 [00:20<00:00,  3.80it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 159, valid acc = 73.870%, loss = 1.1663937334772907
100%|██████████| 391/391 [04:45<00:00,  1.37it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.660%, loss = 1.5101250621973705
100%|██████████| 79/79 [00:21<00:00,  3.72it/s]
epoch = 160, valid acc = 73.880%, loss = 1.1565887475315528
100%|██████████| 391/391 [03:55<00:00,  1.66it/s]
train acc = 78.808%, loss = 1.4890794470487043
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 161, valid acc = 73.960%, loss = 1.153921889353402
100%|██████████| 391/391 [03:26<00:00,  1.90it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 78.976%, loss = 1.4809801377298888
100%|██████████| 79/79 [00:11<00:00,  6.62it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 162, valid acc = 74.070%, loss = 1.1470219130757489
100%|██████████| 391/391 [03:11<00:00,  2.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 79.366%, loss = 1.4545579115143212
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 163, valid acc = 74.860%, loss = 1.1180777957167807
100%|██████████| 391/391 [03:14<00:00,  2.01it/s]
train acc = 79.638%, loss = 1.4306396250529667
100%|██████████| 79/79 [00:12<00:00,  6.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 164, valid acc = 74.730%, loss = 1.1444818339770353
100%|██████████| 391/391 [03:11<00:00,  2.04it/s]
train acc = 80.282%, loss = 1.3997012434713065
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 165, valid acc = 74.360%, loss = 1.1603803853445416
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 80.416%, loss = 1.3850395842586332
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 166, valid acc = 74.470%, loss = 1.1700186359731457
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.232%, loss = 1.3333709829908502
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 167, valid acc = 75.050%, loss = 1.130967435203021
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 81.780%, loss = 1.3001471576483354
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 168, valid acc = 75.290%, loss = 1.1294746738445909
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 82.142%, loss = 1.2672530886767162
100%|██████████| 79/79 [00:11<00:00,  6.67it/s]
epoch = 169, valid acc = 74.670%, loss = 1.1605575371392165
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 82.940%, loss = 1.2217973871609134
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 170, valid acc = 74.450%, loss = 1.1898338583451282
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.616%, loss = 1.1828027881319871
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
epoch = 171, valid acc = 74.830%, loss = 1.180960683882991
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.252%, loss = 1.1391154945354023
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 172, valid acc = 74.860%, loss = 1.2095271228234978
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.700%, loss = 1.110604347322908
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
epoch = 173, valid acc = 75.010%, loss = 1.220273126529742
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.174%, loss = 1.0665440461824618
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
epoch = 174, valid acc = 75.390%, loss = 1.2261392089385021
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 85.794%, loss = 1.0337354472226195
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
epoch = 175, valid acc = 75.430%, loss = 1.235637154005751
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.370%, loss = 1.001444907596959
100%|██████████| 79/79 [00:11<00:00,  6.62it/s]
epoch = 176, valid acc = 74.610%, loss = 1.2795373578614826
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.522%, loss = 0.9796829170278271
100%|██████████| 79/79 [00:11<00:00,  6.62it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 177, valid acc = 74.730%, loss = 1.2896319031715393
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.836%, loss = 0.9611374244970434
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 178, valid acc = 74.810%, loss = 1.2988380669038506
 82%|████████▏ | 319/391 [02:34<00:34,  2.06it/s]
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 87.068%, loss = 0.9505221317796146
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 179, valid acc = 74.930%, loss = 1.2987019540388374
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 87.036%, loss = 0.9571042862694586
100%|██████████| 79/79 [00:12<00:00,  6.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 180, valid acc = 75.290%, loss = 1.2815543865855736
100%|██████████| 391/391 [03:15<00:00,  2.00it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.870%, loss = 0.9534123097844136
100%|██████████| 79/79 [00:12<00:00,  6.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 181, valid acc = 74.540%, loss = 1.3224521845201902
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 87.098%, loss = 0.9492890955237172
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 182, valid acc = 74.670%, loss = 1.3309398580201064
100%|██████████| 391/391 [03:10<00:00,  2.06it/s]
train acc = 86.606%, loss = 0.9651095315318583
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 183, valid acc = 74.800%, loss = 1.3211404281326486
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.760%, loss = 0.9715309844297522
100%|██████████| 79/79 [00:11<00:00,  6.63it/s]
epoch = 184, valid acc = 75.120%, loss = 1.335520737533328
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.326%, loss = 0.9929814806679631
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
epoch = 185, valid acc = 75.080%, loss = 1.3351384159884876
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 86.188%, loss = 1.0198154731479752
100%|██████████| 79/79 [00:12<00:00,  6.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 186, valid acc = 75.140%, loss = 1.3136785135993474
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 85.706%, loss = 1.041593025712406
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 187, valid acc = 74.630%, loss = 1.3351571657989598
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.290%, loss = 1.0652785621335745
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
epoch = 188, valid acc = 75.040%, loss = 1.2996518408195883
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 84.800%, loss = 1.095178504734088
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 189, valid acc = 75.130%, loss = 1.2962923366812211
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.544%, loss = 1.1168583896763795
100%|██████████| 79/79 [00:12<00:00,  6.58it/s]
epoch = 190, valid acc = 74.800%, loss = 1.2875940497917464
100%|██████████| 391/391 [03:11<00:00,  2.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.162%, loss = 1.136768358595231
100%|██████████| 79/79 [00:11<00:00,  6.63it/s]
epoch = 191, valid acc = 75.220%, loss = 1.2767023791240741
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 83.614%, loss = 1.17477103236996
100%|██████████| 79/79 [00:11<00:00,  6.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 192, valid acc = 74.720%, loss = 1.2852420105209834
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 83.230%, loss = 1.2003093538686747
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 193, valid acc = 75.320%, loss = 1.2226662024666992
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.924%, loss = 1.221838514213367
100%|██████████| 79/79 [00:12<00:00,  6.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 194, valid acc = 75.120%, loss = 1.2209133400192744
100%|██████████| 391/391 [03:16<00:00,  1.99it/s]
train acc = 82.552%, loss = 1.238768658522145
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 195, valid acc = 75.000%, loss = 1.2049811738955825
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.646%, loss = 1.2483739334604014
100%|██████████| 79/79 [00:12<00:00,  6.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 196, valid acc = 75.120%, loss = 1.209381861022756
100%|██████████| 391/391 [03:26<00:00,  1.89it/s]
train acc = 82.154%, loss = 1.2695931194688352
100%|██████████| 79/79 [00:12<00:00,  6.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 197, valid acc = 74.940%, loss = 1.2192937678928617
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.258%, loss = 1.2606404006023846
100%|██████████| 79/79 [00:11<00:00,  6.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 198, valid acc = 75.180%, loss = 1.2163929154601278
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
train acc = 82.092%, loss = 1.2709870311000464
100%|██████████| 79/79 [00:12<00:00,  6.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 199, valid acc = 75.110%, loss = 1.1970132061197787
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.328%, loss = 1.2558192077195247
100%|██████████| 79/79 [00:12<00:00,  6.49it/s]
epoch = 200, valid acc = 75.620%, loss = 1.176487018035937
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 82.344%, loss = 1.2584580394922924
100%|██████████| 79/79 [00:12<00:00,  6.55it/s]
epoch = 201, valid acc = 75.470%, loss = 1.181725726097445
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
train acc = 82.648%, loss = 1.2373363861952291
100%|██████████| 79/79 [00:12<00:00,  6.42it/s]
epoch = 202, valid acc = 75.470%, loss = 1.1736313263072242
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.750%, loss = 1.2273447731571734
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 203, valid acc = 75.700%, loss = 1.1860175841971288
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 82.868%, loss = 1.2162661369499344
100%|██████████| 79/79 [00:12<00:00,  6.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 204, valid acc = 75.400%, loss = 1.1731317948691453
100%|██████████| 391/391 [03:12<00:00,  2.04it/s]
train acc = 83.986%, loss = 1.15884984987776
100%|██████████| 79/79 [00:12<00:00,  6.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 205, valid acc = 75.790%, loss = 1.1906958758076536
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 84.188%, loss = 1.1421209261240557
100%|██████████| 79/79 [00:12<00:00,  6.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 206, valid acc = 75.140%, loss = 1.231682456746886
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 84.602%, loss = 1.1046292894636578
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 207, valid acc = 75.960%, loss = 1.211782468270652
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 85.566%, loss = 1.0562567663619586
100%|██████████| 79/79 [00:12<00:00,  6.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 208, valid acc = 75.440%, loss = 1.2416871695578853
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
train acc = 85.838%, loss = 1.022005300692585
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
epoch = 209, valid acc = 75.810%, loss = 1.2158979071846492
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.550%, loss = 0.984019377195012
100%|██████████| 79/79 [00:11<00:00,  6.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 210, valid acc = 75.480%, loss = 1.238747905326795
100%|██████████| 391/391 [03:16<00:00,  1.99it/s]
train acc = 87.302%, loss = 0.9369751227176403
100%|██████████| 79/79 [00:12<00:00,  6.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 211, valid acc = 75.520%, loss = 1.269252433807035
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
train acc = 88.044%, loss = 0.8969368862221613
100%|██████████| 79/79 [00:12<00:00,  6.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 212, valid acc = 75.250%, loss = 1.29635628793813
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
train acc = 88.552%, loss = 0.8501449200655798
100%|██████████| 79/79 [00:12<00:00,  6.57it/s]
epoch = 213, valid acc = 75.160%, loss = 1.312730856334107
100%|██████████| 391/391 [03:13<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 89.128%, loss = 0.8196710208645257
100%|██████████| 79/79 [00:12<00:00,  6.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 214, valid acc = 75.670%, loss = 1.3123768258698378
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
train acc = 89.824%, loss = 0.7810705409330481
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 215, valid acc = 75.750%, loss = 1.3378497663932511
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 90.054%, loss = 0.7592855453338769
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 216, valid acc = 75.140%, loss = 1.3778831151467335
100%|██████████| 391/391 [03:11<00:00,  2.05it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.514%, loss = 0.7348553946103586
100%|██████████| 79/79 [00:11<00:00,  6.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 217, valid acc = 75.060%, loss = 1.4105089148388634
100%|██████████| 391/391 [03:10<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.782%, loss = 0.7147462589051717
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
epoch = 218, valid acc = 74.720%, loss = 1.4321710256081592
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 90.714%, loss = 0.713776810547275
100%|██████████| 79/79 [00:11<00:00,  6.63it/s]
epoch = 219, valid acc = 75.510%, loss = 1.3961331459540356
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.876%, loss = 0.7063642440701995
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 220, valid acc = 75.000%, loss = 1.4379620031465459
100%|██████████| 391/391 [03:14<00:00,  2.01it/s]
train acc = 91.124%, loss = 0.6951159863825649
100%|██████████| 79/79 [00:12<00:00,  6.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 221, valid acc = 75.430%, loss = 1.4394663919376423
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 90.824%, loss = 0.7096468377906038
100%|██████████| 79/79 [00:12<00:00,  6.34it/s]
epoch = 222, valid acc = 75.050%, loss = 1.4485547097423408
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.712%, loss = 0.7180800011853123
100%|██████████| 79/79 [00:12<00:00,  6.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 223, valid acc = 74.740%, loss = 1.4495541302463677
100%|██████████| 391/391 [03:13<00:00,  2.02it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 90.628%, loss = 0.7243357188713825
100%|██████████| 79/79 [00:12<00:00,  6.30it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 224, valid acc = 75.200%, loss = 1.4086205559440805
100%|██████████| 391/391 [03:14<00:00,  2.01it/s]
train acc = 90.192%, loss = 0.75091017374907
100%|██████████| 79/79 [00:12<00:00,  6.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 225, valid acc = 75.270%, loss = 1.4455406062210663
100%|██████████| 391/391 [03:14<00:00,  2.01it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 89.880%, loss = 0.7642111407826319
100%|██████████| 79/79 [00:12<00:00,  6.21it/s]
epoch = 226, valid acc = 75.080%, loss = 1.4449767130839675
100%|██████████| 391/391 [03:11<00:00,  2.04it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 89.414%, loss = 0.791460419402403
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
epoch = 227, valid acc = 75.220%, loss = 1.4444586898707137
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 88.954%, loss = 0.829657779553967
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 228, valid acc = 75.200%, loss = 1.4243354246586184
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 88.680%, loss = 0.8465368262947063
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 229, valid acc = 75.100%, loss = 1.4167226718950876
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 88.190%, loss = 0.8841220984221114
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 230, valid acc = 74.830%, loss = 1.4230802149712285
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 87.372%, loss = 0.9333783732655713
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 231, valid acc = 75.630%, loss = 1.3632010146032405
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 87.004%, loss = 0.949613904556655
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 232, valid acc = 75.200%, loss = 1.3934798353834996
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.776%, loss = 0.9746479131376652
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 233, valid acc = 75.450%, loss = 1.3492311812654327
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.366%, loss = 0.9956191939771023
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 234, valid acc = 75.860%, loss = 1.3109310495702526
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 85.752%, loss = 1.0201271792202045
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 235, valid acc = 75.130%, loss = 1.3423070598252211
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
train acc = 85.606%, loss = 1.0369647235211814
100%|██████████| 79/79 [00:11<00:00,  6.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 236, valid acc = 75.300%, loss = 1.2936392683017104
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.488%, loss = 1.0506194880246507
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
epoch = 237, valid acc = 75.340%, loss = 1.299656872507892
100%|██████████| 391/391 [03:09<00:00,  2.06it/s]
train acc = 85.552%, loss = 1.0597387803789904
100%|██████████| 79/79 [00:12<00:00,  6.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 238, valid acc = 75.840%, loss = 1.263087908678417
100%|██████████| 391/391 [03:12<00:00,  2.03it/s]
train acc = 85.252%, loss = 1.0652076169048124
100%|██████████| 79/79 [00:11<00:00,  6.62it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 239, valid acc = 75.750%, loss = 1.2719533730156813
100%|██████████| 391/391 [03:48<00:00,  1.71it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.364%, loss = 1.060642236638862
100%|██████████| 79/79 [00:14<00:00,  5.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 240, valid acc = 75.350%, loss = 1.2576196638843682
100%|██████████| 391/391 [03:21<00:00,  1.94it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.508%, loss = 1.0553218574475145
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 241, valid acc = 75.840%, loss = 1.261670604536805
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.656%, loss = 1.0370259759066356
100%|██████████| 79/79 [00:11<00:00,  6.67it/s]
epoch = 242, valid acc = 75.550%, loss = 1.2647303365453888
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 85.858%, loss = 1.0357819807803845
100%|██████████| 79/79 [00:11<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 243, valid acc = 76.300%, loss = 1.2352659015715877
100%|██████████| 391/391 [03:09<00:00,  2.07it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 86.484%, loss = 0.9935869144661652
100%|██████████| 79/79 [00:11<00:00,  6.59it/s]
epoch = 244, valid acc = 75.960%, loss = 1.2566227196138116
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 86.772%, loss = 0.9813652735232087
100%|██████████| 79/79 [00:12<00:00,  6.30it/s]
epoch = 245, valid acc = 75.440%, loss = 1.2681946965712536
100%|██████████| 391/391 [03:16<00:00,  1.99it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 87.444%, loss = 0.9287419857271492
100%|██████████| 79/79 [00:11<00:00,  6.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 246, valid acc = 75.460%, loss = 1.2893925973131686
100%|██████████| 391/391 [03:15<00:00,  2.00it/s]
train acc = 87.786%, loss = 0.9078031612174285
100%|██████████| 79/79 [00:12<00:00,  6.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 247, valid acc = 75.300%, loss = 1.2973330639585663
100%|██████████| 391/391 [03:17<00:00,  1.98it/s]
  0%|          | 0/79 [00:00<?, ?it/s]train acc = 88.410%, loss = 0.8650390679574074
100%|██████████| 79/79 [00:12<00:00,  6.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 248, valid acc = 75.370%, loss = 1.3317630698409262
100%|██████████| 391/391 [03:16<00:00,  1.99it/s]
train acc = 88.910%, loss = 0.8235288330203737
100%|██████████| 79/79 [00:12<00:00,  6.58it/s]
epoch = 249, valid acc = 75.560%, loss = 1.3275073517726947
100%|██████████| 391/391 [03:10<00:00,  2.05it/s]
train acc = 90.024%, loss = 0.7663024105226902
100%|██████████| 79/79 [00:11<00:00,  6.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]epoch = 250, valid acc = 75.450%, loss = 1.369174149971974
```

