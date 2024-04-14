# Deit_repetition
repete deit model

## Background 

ViT

- [ ] 归纳偏置?
- [ ] 训练的时候用一个隐藏层的MLP，Finetune的时候用一个线性层

class token是embedded得到

1D position embedding 加入到patch embedding

layer norm在每一个block前



ViT的Pytorch代码：

[FrancescoSaverioZuppichini/ViT: Implementing Vi(sion)T(transformer) (github.com)](https://github.com/FrancescoSaverioZuppichini/ViT)

[vit-pytorch/vit_pytorch/vit.py at main · lucidrains/vit-pytorch (github.com)](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)



蒸馏

- [x] [知识蒸馏技术（教师学生模型） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/442457531)



散度和交叉熵

- [x] [Kullback-Leibler(KL)散度介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/100676922)(这个讲得还挺有意思的，虽然Loss计算很简单)

- [x] [交叉熵和KL散度有什么区别？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/292434104)



### 问题

- [ ] position embedding在Transformer之前添加就可以了吗？

- [ ] 为什么class token和distillation token要分开，不能都用class token计算loss吗？

- [ ] 硬蒸馏和软蒸馏的区别？（散度和交叉熵的区别？）

  

- [ ] 怎么初始化参数

  initialize the weights with a truncated normal distribution

- [ ] 数据和模型怎么放到GPU上面跑

- [ ] class token和position从embedding为什么是随机的

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



### 依赖

```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```