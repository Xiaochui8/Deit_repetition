from model.vit import ViT


def get_model(model_name, args):
    if model_name == 'vit':
        return ViT(args['in_channels'], args['patch_size'], args['embed_size'], args['image_size'], args['depth'], args['num_heads'],
                   args['drop_p'], args['forward_expansion'], args['forward_drop_p'], args['num_classes'], args['if_cls'],
                   args['if_dis'])