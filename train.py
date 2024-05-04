from data.data_load import get_data
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from tools.ema import EMA
from torch.cuda.amp import autocast, GradScaler



def initialize_model_weights(model, mean=0.0, std=0.02):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.trunc_normal_(param, mean=mean, std=std)


def load_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']

    return model, optimizer, epoch, train_loss, train_acc

def save_checkpoint(checkpoint_path, model, scheduler, optimizer, epoch, train_loss, train_acc):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc
    }
    torch.save(checkpoint, checkpoint_path)


def train(data_args, train_args, model_name, model, device):
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
    model_save_name = train_args['model_save_name']


    train_dataloader, val_dataloader = get_data(path, resolution, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5E-5)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args['cosine_max_epochs'], eta_min=train_args['eta_min'])
    ema = EMA(model, decay=0.999)
    ema.register()
    scaler = GradScaler()


    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    last_epoch = -1
    start_epoch = last_epoch + 1


    # ========================================================load checkpoint========================================================

    if start_epoch == 0:
        initialize_model_weights(model)
    else:
        checkpoint_load_path = checkpoint_path + "/" + model_name + "_{}.pth".format(last_epoch)
        load_checkpoint(checkpoint_load_path, model, optimizer, scheduler)




    for epoch in range(start_epoch, start_epoch + epochs):
        # model.half()
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # ========================================================train========================================================
        for idx, (inputs, label) in enumerate(tqdm(train_dataloader)):
            # transfer data to device
            inputs = inputs.to(device)
            label = label.to(device)


            # forward
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, label)

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update()

            train_epoch_loss.append(loss.item())

            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]

        scheduler.step()
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))

        # ========================================================save checkpoint========================================================
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_save_path = checkpoint_path + "/" + model_name + "_{}.pth".format(epoch)
            save_checkpoint(checkpoint_save_path, model, scheduler, optimizer, epoch, np.average(train_epoch_loss), 100 * acc / nums)

        # ========================================================validate========================================================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            ema.apply_shadow()

            for idx, (inputs, label) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(device) # .to(torch.float)
                label = label.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                train_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)
            ema.restore()

            print("epoch = {}, valid acc = {:.3f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))


def distill(data_args, train_args, student_name, student_model, teacher_name, teacher_model, device):

    # data arguments
    path = data_args['datasets_path']
    resolution = data_args['resolution']
    batch_size = data_args['batch_size']

    # training arguments
    lr = train_args['lr']
    epochs = train_args['num_epochs']
    checkpoint_interval = train_args['checkpoint_interval']
    checkpoint_path = train_args['checkpoint_path']
    tau = train_args['tau']
    lambd = train_args['lambda']



    train_dataloader, val_dataloader = get_data(path, resolution, batch_size)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5E-5)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args['cosine_max_epochs'], eta_min=train_args['eta_min'])
    ema = EMA(student_model, decay = 0.999)
    ema.register()


    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    last_epoch = -1
    start_epoch = last_epoch + 1


    # ========================================================load checkpoint========================================================

    if start_epoch == 0:
        initialize_model_weights(student_model)
    else:
        checkpoint_load_path = checkpoint_path + "/" + student_name + "_dis" + "_{}.pth".format(last_epoch)
        load_checkpoint(checkpoint_load_path, student_model, optimizer, scheduler)




    for epoch in range(start_epoch, start_epoch + epochs):
        student_model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # ========================================================train========================================================
        for idx, (inputs, label) in enumerate(tqdm(train_dataloader)):
            # transfer data to device
            inputs = inputs.to(device)
            label = label.to(device)


            # forward
            outputs = student_model(inputs)
            loss = (1 - lambd) * criterion1(outputs, label)

            outputs_teacher = teacher_model(inputs)
            loss += lambd * tau * tau * nn.functional.kl_div((outputs / tau).softmax(-1).log(), (outputs_teacher / tau).softmax(-1), reduction='batchmean')

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

            train_epoch_loss.append(loss.item())

            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        scheduler.step()
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))

        # ========================================================save checkpoint========================================================
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_save_path = checkpoint_path + "/" + student_name + "_dis" + "_{}.pth".format(epoch)
            save_checkpoint(checkpoint_save_path, student_model, scheduler, optimizer, epoch, np.average(train_epoch_loss), 100 * acc / nums)

        # ========================================================validate========================================================
        with torch.no_grad():
            student_model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            ema.apply_shadow()

            for idx, (inputs, label) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(device)  # .to(torch.float)
                label = label.to(device)
                outputs = student_model(inputs)
                loss = criterion1(outputs, label)
                val_epoch_loss.append(loss.item())

                train_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)
            ema.restore()

            print("epoch = {}, valid acc = {:.3f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))