from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import visdom
import os
import time
from os import listdir
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import utils, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from networks.styleTnet import StyleTransferNet
import argparse

parser = argparse.ArgumentParser()
# Basic options

parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()



# Some utilities
class VisdomLine():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, x, y):
        if self.win is None:
            self.win = self.vis.line(X=x, Y=y, opts=self.opts)
        else:
            self.vis.line(X=x, Y=y, opts=self.opts, win=self.win)

class VisdomImage():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, image):
        if self.win is None:
            self.win = self.vis.image(image, opts=self.opts)
        else:
            self.vis.image(image, opts=self.opts, win=self.win)

def LearningRateScheduler(optimizer, epoch, lr_decay=0.1, lr_decay_step=10):
    if epoch % lr_decay_step:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    print('Learning rate is decreased by %f' % (lr_decay))

    return optimizer



# For data loading
class DataManager(Dataset):
    def __init__(self, path_content, path_style, random_crop=True):
        self.path_content = path_content
        self.path_style = path_style

        # Preprocessing for imagenet pre-trained network
        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )

        # Convert pre-processed images to original images
        self.restore = transforms.Compose(
            [
                transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                     std=[4.367, 4.464, 4.444]),
            ]
        )

        self.list_content = listdir(self.path_content)
        self.list_style = listdir(self.path_style)

        self.num_content = len(self.list_content)
        self.num_style = len(self.list_style)

        assert self.num_content > 0
        assert self.num_style > 0

        self.num = min(self.num_content, self.num_style)

        print('Content root : %s' % (self.path_content))
        print('Style root : %s' % (self.path_style))
        print('Number of content images : %d' % (self.num_content))
        print('Number of style images : %d' % (self.num_style))
        print('Dataset size : %d' % (self.num))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        path_to_con = self.path_content + '/' + self.list_content[idx]
        path_to_sty = self.path_style + '/' + self.list_style[idx]

        img_con = Image.open(path_to_con)
        img_con = self.transform(img_con)

        img_sty = Image.open(path_to_sty)
        img_sty = self.transform(img_sty)

        sample = {'content': img_con, 'style': img_sty}

        return sample



def train():
    gc.disable()

    # Parameters
    path_snapshot = 'snapshots'
    path_content = 'dataset/train/content'
    path_style = 'dataset/train/style'

    if not os.path.exists(path_snapshot):
        os.makedirs(path_snapshot)

    batch_size = 16
    weight_decay = 1.0e-5
    num_epoch = 1001
    lr_init = 0.001
    lr_decay_step = num_epoch/2
    momentum = 0.9
    device_ids = [0, 1]
    w_style = 1.01

    # Data loader
    dm = DataManager(path_content, path_style, random_crop=True)
    dl = DataLoader(dm, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    num_train = dm.num
    num_batch = np.ceil(num_train / batch_size)
    loss_train_avg = np.zeros(num_epoch)

    net = StyleTransferNet(w_style)

    """
            Defining  optimizer  here to train only the decoder network
        """
    #    optimizer = None
    optimizer = torch.optim.Adam(net.decoder.parameters(), lr=args.lr)


    net = nn.DataParallel(net.cuda(), device_ids=device_ids)
    net.train()



    # For visualization
    vis = visdom.Visdom()

    vis_loss = VisdomLine(vis, dict(title='Training Loss', markers=True))
    vis_image = VisdomImage(vis, dict(title='Content / Style / Result'))

    # Start training
    for epoch in range(0, num_epoch):
        running_loss_train = 0
        np.random.shuffle(dl.dataset.list_style)

        for i, data in enumerate(dl, 0):
            img_con = data['content']
            img_sty = data['style']

            img_con = Variable(img_con, requires_grad=False).cuda()
            img_sty = Variable(img_sty, requires_grad=False).cuda()

            optimizer.zero_grad()

            loss, img_result = net(img_con, img_sty)

            loss = torch.mean(loss)
            loss.backward()

            optimizer.step()

            running_loss_train += loss

            print('[%s] Epoch %3d / %3d, Batch %5d / %5d, Loss = %12.8f' %
                  (str(datetime.now())[:-3], epoch + 1, num_epoch,
                   i + 1, num_batch, loss))


        loss_train_avg[epoch] = running_loss_train / num_batch

        print('[%s] Epoch %3d / %3d, Avg Loss = %12.8f' % \
              (str(datetime.now())[:-3], epoch + 1, num_epoch,
               loss_train_avg[epoch]))

        optimizer = LearningRateScheduler(optimizer, epoch + 1, lr_decay_step=lr_decay_step)

        # Display using visdom
        vis_loss.Update(np.arange(epoch + 1) + 1, loss_train_avg[0:epoch + 1])

        img_cat = torch.cat((img_con, img_sty, img_result), dim=3)
        img_cat = torch.unbind(img_cat, dim=0)
        img_cat = torch.cat(img_cat, dim=1)
        img_cat = dm.restore(img_cat.data.cpu())
        vis_image.Update(torch.clamp(img_cat, 0, 1))

        # Snapshot
        if (epoch % 100) == 0:
            torch.save(net, '%s/epoch_%06d.pth' % (path_snapshot, epoch + 1))

        gc_collected = gc.collect()
        gc.disable()

    print('Training finished.')



if __name__ == '__main__':
    train()