import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal


class Encoder(nn.Module):

    def __init__(self,batchNorm=True):
        super(Encoder,self).__init__()


        self.encoder =  nn.Sequential(
                                        nn.Conv2d(3, 3, (1, 1)),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(3, 64, (3, 3)),
                                        nn.ReLU(),  # relu1-1
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(64, 64, (3, 3)),
                                        nn.ReLU(),  # relu1-2
                                        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(64, 128, (3, 3)),
                                        nn.ReLU(),  # relu2-1
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(128, 128, (3, 3)),
                                        nn.ReLU(),  # relu2-2
                                        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(128, 256, (3, 3)),
                                        nn.ReLU(),  # relu3-1
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(256, 256, (3, 3)),
                                        nn.ReLU(),  # relu3-2
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(256, 256, (3, 3)),
                                        nn.ReLU(),  # relu3-3
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(256, 256, (3, 3)),
                                        nn.ReLU(),  # relu3-4
                                        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(256, 512, (3, 3)),
                                        nn.ReLU(),  # relu4-1, this is the last layer used
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu4-2
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu4-3
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu4-4
                                        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu5-1
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu5-2
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU(),  # relu5-3
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(512, 512, (3, 3)),
                                        nn.ReLU()  # relu5-4
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
        out = self.encoder(x)

        return out


def encoderNet(pretrained=True):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = Encoder(batchNorm=False)
    if pretrained is True:
        model.encoder.load_state_dict(torch.load('models/vgg_normalised.pth'))
    return model


