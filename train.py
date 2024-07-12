import os
import sys
import pandas as pd

import numpy as np
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import v3d_io
from main import SwinTransformerClassify


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


class MyDataset(Dataset):
    def __init__(self, txt_file, root_dir):
        self.labels_frame = pd.read_csv(txt_file, header=None, names=['image', 'label'], sep=',')
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        img = v3d_io.load_v3d_raw_img_file(img_name)
        img_data = img["data"]
        img_data = torch.from_numpy(img_data).cuda()
        img_data = img_data.permute(3, 0, 1, 2)
        img_data = img_data.float()

        label = self.labels_frame.iloc[idx, 1]
        label = torch.tensor(label).long().cuda()  # assuming that the label is an integer

        return img_data, label


def main():
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    batch_size = 1
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 500  # max traning epoch
    cont_training = False  # if continue training

    model_dir = "models/"

    logger = Logger("logs/")

    traindataset = MyDataset(txt_file='train.txt', root_dir='Train/')
    train_loader = DataLoader(traindataset, batch_size=1, shuffle=True)

    valdataset = MyDataset(txt_file='val.txt', root_dir='Val/')
    val_loader = DataLoader(valdataset, batch_size=1, shuffle=True)

    model = SwinTransformerClassify()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if cont_training and os.path.exists(model_dir):
        checkpoints = os.listdir(model_dir)
        if checkpoints:
            # 使用natsort进行自然排序，然后选择最后一个checkpoint
            latest_checkpoint = natsorted(checkpoints)[-1]
            checkpoint_path = os.path.join(model_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch'] + 1
            print('Model: {} loaded!'.format(latest_checkpoint))
        else:
            epoch_start = 0
    else:
        epoch_start = 0

    for epoch in range(epoch_start, max_epoch):
        for data, target in train_loader:
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            output = model(data)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.write('Epoch: {}, Loss: {}\n'.format(epoch, loss.item()))

        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        #
        # state = {
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }
        # torch.save(state, os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch)))
        # logger.write('Model saved! {}\n'.format(epoch))


if __name__ == '__main__':
    main()
