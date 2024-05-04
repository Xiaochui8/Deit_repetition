from model.vit import ViT
from model.temp_model import ViT2
from model.ResNet import ResNet18_1, ResNet18_2
import torch

def get_model(args):
    if args['model_name'] == 'vit':
        return ViT(args['in_channels'], args['patch_size'], args['embed_size'], args['image_size'], args['depth'], args['num_heads'],
                   args['drop_p'], args['forward_expansion'], args['forward_drop_p'], args['num_classes'], args['if_cls'],
                   args['if_dis'])
    if args['model_name'] == 'vit2':
        return ViT2(image_size=args['image_size'], patch_size=args['patch_size'], num_classes=args['num_classes'], dim=args['dim'],
                    depth=args['depth'], heads=args['heads'], mlp_dim=args['mlp_dim'], pool='cls', channels=args['channels'],
                    dim_head=args['dim_head'], dropout=args['dropout'], emb_dropout=args['emb_dropout'])
    if args['model_name'] == 'resnet18_1':
        return ResNet18_1()
    if args['model_name'] == 'resnet18_2':
        return ResNet18_2()


def load_teacher_model(checkpoint_path = None, model = None, optimizer = None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']

    return model, optimizer, epoch, train_loss, train_acc

def get_teacher_model(args):
    if args['model_name'] == 'resnet18_1':
        teacher_model = ResNet18_1()
        load_teacher_model(model=teacher_model, checkpoint_path = 'checkpoint/resnet18_1_99.pth')
        return teacher_model
    if args['model_name'] == 'resnet18_2':
        teacher_model = ResNet18_2()
        load_teacher_model(model=teacher_model,  checkpoint_path = 'checkpoint/resnet18_2_59.pth')
        return teacher_model