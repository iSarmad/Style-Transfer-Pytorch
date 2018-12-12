
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
parser.add_argument('--modelpath', type=str, default="snapshots/epoch_000901.pth")
parser.add_argument('--contentpath', type=str, default="dataset/test/content")
parser.add_argument('--stylepath', type=str, default="dataset/test/style")
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

class DataManager(Dataset):
    def __init__(self, path_content, path_style, center_crop=True):
        self.path_content = path_content
        self.path_style = path_style

        # Preprocessing for imagenet pre-trained network
        if center_crop:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop((280, 280)),
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

def test ():

    # data
    dm = DataManager(args.contentpath,args.stylepath)
    dl = DataLoader(dm,batch_size=10,shuffle=False,num_workers=4,pin_memory=True)

   # devices = [0]
   # net = StyleTransferNet()
    #net = nn.DataParallel(net.cuda(),device_ids=devices)
    net = torch.load(args.modelpath)
    #loader = loader.module.state_dict()
   # net = net.load_state_dict(loader)
    net.eval()

    lista = [0.0 , 0.5 ,1.0]
   # content = Variable()
    for alpha in lista:

        for i, data in enumerate(dl, 0):
            img_con = data['content']
            img_sty = data['style']

            img_con = Variable(img_con, requires_grad=False).cuda()
            img_sty = Variable(img_sty, requires_grad=False).cuda()

            img_result = net(img_con, img_sty,alpha=alpha)

            vis = visdom.Visdom()
            vis_image = VisdomImage(vis, dict(title='Content / Style / Result'))


            img_cat = torch.cat((img_con, img_sty, img_result), dim=3)
            img_cat = torch.unbind(img_cat, dim=0)
            img_cat = torch.cat(img_cat, dim=1)
            img_cat = dm.restore(img_cat.data.cpu())
            vis_image.Update(torch.clamp(img_cat, 0, 1))


if __name__=="__main__":
    test()