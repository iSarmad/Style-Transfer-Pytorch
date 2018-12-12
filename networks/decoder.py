import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal


class Decoder(nn.Module):

    def __init__(self,batchNorm=True):
        super(Decoder,self).__init__()

        self.batchNorm = batchNorm
        self.decoder =  nn.Sequential(
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(512, 256, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),#ESPCN(upscale_factor=2,in_ch=256),#

                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 128, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),# ESPCN(upscale_factor=2, in_ch=128),  #

                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 128, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 64, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),#ESPCN(upscale_factor=2, in_ch=64),  #

                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 64, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 3, (3, 3)),
                    )



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data) #initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_() # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x):
        # Encoder section
        out = self.decoder(x)

        return out


# class ESPCN(nn.Module):
#     def __init__(self, in_ch=1, upscale_factor= 4):
#         super(ESPCN, self).__init__()
#         self.name = 'ESPCN'
#         self.upfactor = upscale_factor
#
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(in_ch, 64, 5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, upscale_factor ** 2, 3, stride=1, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.pixel_shuffle(self.conv3(x))
#         return x


def decoderNet(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = Decoder(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def decoderNet_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = Decoder(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model