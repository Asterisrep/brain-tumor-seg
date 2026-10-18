import torch
import torch.nn as nn


class ConvBLock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),    
        )


    def forward(self, x):
        return self.block(x)



class UNet2D(nn.Module):

    def __init__(self, in_ch = 4, out_ch = 4, base = 32, levels = 4):
        super().__init__()

        self.levels = levels
        chs = [base*2**i for i in range(levels + 1)]

        ## Encoder ##
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.enc.append(ConvBLock(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)


        ## Bottleneck ##
        self.bottleneck = ConvBLock(chs[-2], chs[-1])


        ## Decoder ##
        self.up = nn.ModuleList()   
        self.dec = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2))
            self.dec.append(ConvBLock(c * 2, c))

        self.head = nn.Conv2d(chs[0], out_ch, kernel_size=1)


    def forward(self, x):
        div = 2 ** self.levels
        assert x.shape[-1] % div == 0 and x.shape[-2] % div == 0, \
            f"H and W must be divisible by {div}, got {tuple(x.shape[-2:])}"


        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec in zip(self.up, self.dec):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)
