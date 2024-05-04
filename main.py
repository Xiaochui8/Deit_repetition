import json
import argparse
import torch
from train import train, distill
from model.get_model import get_model, get_teacher_model

if __name__ == '__main__':
    with open('config.json', 'r') as file:
        config = json.load(file)

    arguments = argparse.ArgumentParser()
    arguments.add_argument("-task", help="the task to do")
    args = arguments.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.task == 'train_vit':
        config_model = config['vit_args']
        train(config['data_args'], config['train_args'], config_model['model_name'], get_model(config_model).to(device), device)
    if args.task == 'distill_vit':
        config_student_model = config['vit_args']
        config_teacher_model = config['resnet18_2_args']
        distill(config['data_args'], config['train_args'], config_student_model['model_name'], get_model(config_student_model).to(device), config_teacher_model['model_name'], get_teacher_model(config_teacher_model).to(device), device)