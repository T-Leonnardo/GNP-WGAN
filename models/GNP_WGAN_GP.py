import torch.nn as nn
import torch
import torch.nn.functional as F

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm:
            self.d_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, mid_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.d_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        return self.d_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels,out_channels,norm=True):
        super().__init__()
        if norm:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.GroupNorm(32,in_channels//2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class GNP_WGAN_GP(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(GNP_WGAN_GP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = (Down(n_channels, 32, norm=False))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))
        self.down5 = (Down(256, 256))
        self.down6 = (Down(256, 512))
        self.down7 = (Down(512, 512))
        self.up1 = Up(512,512)
        self.up2 = Up(1024,256)
        self.up3 = Up(512,256)
        self.up4 = Up(512,128)
        self.up5 = Up(256,64)
        self.up6 = Up(128,32)
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)
        return logits

class PatchGNP_WGAN_Discriminator(nn.Module):
    def __init__(self, input_channels=1, ndf=32):
        super(PatchGNP_WGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1,bias=False)
        self.act = nn.SiLU(True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn2 = nn.GroupNorm(32, ndf*2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn3 = nn.GroupNorm(32, ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn4 = nn.GroupNorm(32, ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn5 = nn.GroupNorm(32, ndf * 16)
        # Final layer
        self.conv6 = nn.Conv2d(ndf * 16, 1, kernel_size=1)
        self.act_out = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.act_out(x)
        return x