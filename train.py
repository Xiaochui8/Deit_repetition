from data.data_load import get_data
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

# TODO: 初始化model

def initialize_model_weights(model, mean=0.0, std=0.02):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.trunc_normal_(param, mean=mean, std=std)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # train_loss = checkpoint['train_loss']
    # train_acc = checkpoint['train_acc']

    return model, optimizer, epoch


def train(data_args, train_args, model, device):
    # data arguments
    path = data_args['datasets_path']
    resolution = data_args['resolution']
    batch_size = data_args['batch_size']

    # training arguments
    lr = train_args['lr']
    momentum = train_args['momentum']
    epochs = train_args['num_epochs']
    checkpoint_interval = train_args['checkpoint_interval']
    checkpoint_path = train_args['checkpoint_path']
    model_save_path = train_args['model_save_path']


    train_dataloader, val_dataloader = get_data(path, resolution, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    last_epoch = 4
    start_epoch = last_epoch + 1


    # ========================================================load checkpoint========================================================

    if start_epoch == 0:
        initialize_model_weights(model)
    else:
        load_checkpoint(checkpoint_path + '/' + "epoch_{}.pth".format(last_epoch), model, optimizer)
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (inputs, label) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(device)  # .to(torch.float)
                label = label.to(device)
                outputs = model(inputs)
                outputs = outputs[:, 0, :].squeeze()
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)
            print("epoch = {}, valid acc = {:.3f}%, loss = {}".format(last_epoch, 100 * acc / nums,
                                                                      np.average(val_epoch_loss)))



    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # ========================================================train========================================================
        for idx, (inputs, label) in enumerate(tqdm(train_dataloader)):

            inputs = inputs.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs[:,0,:].squeeze()
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))

        # ========================================================save checkpoint========================================================
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }
            checkpoint_path_fianl = checkpoint_path +'/' + "epoch_{}.pth".format(epoch)
            torch.save(checkpoint, checkpoint_path_fianl)

        # ========================================================validate========================================================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (inputs, label) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(device)  # .to(torch.float)
                label = label.to(device)
                outputs = model(inputs)
                outputs = outputs[:, 0, :].squeeze()
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print("epoch = {}, valid acc = {:.3f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))