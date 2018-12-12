from .decoder import *
from .encoderNet import *

def crop_like(input, target):
    # if input size is different than target size, crop input size to as small as target size
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)] #[batch, channels, H, W]

def std(x,tol= 1e-5):
    s = x.size()
    n, c = s[:2]
    var = tol + x.view(n, c, -1).var(dim=2)
    std = var.sqrt().view(n, c, 1, 1)
    return std

def mean(x):
    s = x.size()
    n, c = s[:2]
    mean = x.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean

class StyleTransferNet(nn.Module):
    def __init__(self, w_style=10.0):
        super(StyleTransferNet, self).__init__()

        self.w_style = w_style
        encoder = encoderNet()
        decoder = decoderNet()
        enc_layers  = list(encoder.encoder.children())
        self.enc_1 =  nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.mse_loss = nn.MSELoss()
        self.decoder = decoder.decoder

        dec_layers = list(decoder.decoder.children())
        self.dec_1 = nn.Sequential(*dec_layers[:4])
        self.dec_2 = nn.Sequential(*dec_layers[4:17])
        self.dec_3 = nn.Sequential(*dec_layers[17:24])
        self.dec_4 = nn.Sequential(*dec_layers[24:])



        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False



    def encoder_stages(self, x):
        out = [x]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            out.append(func(out[-1]))
        return out[1:]

    def encode(self, x):
        for i in range(4):
            x = getattr(self, 'enc_{:d}'.format(i + 1))(x)
        return x

    def loss_content(self, x, y):
        assert (x.size() == y.size())
        assert (y.requires_grad is False)
        return self.mse_loss(x, y)


    def loss_style(self, x, y):
        assert (x.size() == y.size())
        assert (y.requires_grad is False)

        mean_x = mean(x)
        std_x = std(x)

        mean_y = mean(y)
        std_y = std(y)

        loss = self.mse_loss(mean_x, mean_y) + self.mse_loss(std_x, std_y)

        return loss

    def AdaINLayer(self, x, y):
        Bx, Cx, Hx, Wx = x.shape
        By, Cy, Hy, Wy = y.shape

        assert Bx == By
        assert Cx == Cy


        s = x.size()

        mean_content = mean(x)
        std_content = std(x)

        mean_style = mean(y)
        std_style = std(y)

        "ADIN Layer Formula"
        content_norm = (x - mean_content.expand(s)) / std_content.expand(s)
        output= std_style.expand(s) * content_norm  + mean_style.expand(s)
        return output

    def forward(self, content, style,alpha=1.0):
       # B, C, H, W = x.shape

        """

        forward process, content loss , images

        """

        if self.training is  True:
            assert 0 <= alpha <= 1
            style_feats = self.encoder_stages(style)

            content_feat = self.encode(content)

            out_adin = self.AdaINLayer(content_feat, style_feats[-1])

            out_adin = alpha * out_adin + (1 - alpha) * content_feat

            dec_out = self.decoder(out_adin)

            # find all the intermediate features

            dec_stages = self.encoder_stages(dec_out)

            Sloss = self.loss_style(dec_stages[0], style_feats[0])

            # content loss calculation
            Closs = self.loss_content(dec_stages[-1], out_adin)
            for i in range(1, 4):
            # loss of style
                Sloss += self.loss_style(dec_stages[i], style_feats[i])
            # total loss using wstyle
            loss = Closs + Sloss * self.w_style

            return loss, dec_out
        else:
            assert 0 <= alpha <= 1
            style_feats = self.encoder_stages(style)
            content_feat = self.encode(content)
            out_adin = self.AdaINLayer(content_feat, style_feats[-1])
            out_adin = alpha * out_adin + (1 - alpha) * content_feat

            dec_out = self.decoder(out_adin)

            return dec_out
