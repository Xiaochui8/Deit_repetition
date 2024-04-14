import json
import argparse
import torch
from train import train
from model.get_model import get_model

if __name__ == '__main__':
    with open('config.json', 'r') as file:
        config = json.load(file)

    arguments = argparse.ArgumentParser()
    arguments.add_argument("-task", help="the task to do")
    args = arguments.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.task == 'train_vit':
        train(config['data_args'], config['train_args'], get_model('vit', config['vit_args']).to(device), device)