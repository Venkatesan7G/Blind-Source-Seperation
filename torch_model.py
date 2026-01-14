import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        return x


class TFMaskUNet(nn.Module):
    """
    Input:  (B, 5, F, T)
    Output: (B, 2, F, T) masks with softmax across source dim (sum=1 per TF-bin)
    """
    def __init__(self, in_ch=5, base=32, n_src=2):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bott = ConvBlock(base * 2, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2a = nn.Conv2d(base * 4, base * 2, kernel_size=2, padding=0)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1a = nn.Conv2d(base * 2, base, kernel_size=2, padding=0)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, n_src, kernel_size=1)

    def forward(self, x):
        # x: (B,5,F,T)
        c1 = self.enc1(x)          # (B,base,F,T)
        p1 = self.pool1(c1)        # (B,base,F/2,T/2)

        c2 = self.enc2(p1)         # (B,2base,F/2,T/2)
        p2 = self.pool2(c2)        # (B,2base,F/4,T/4)

        b = self.bott(p2)          # (B,4base,F/4,T/4)

        u2 = self.up2(b)           # (B,4base,F/2,T/2)
        u2 = F.relu(self.dec2a(u2))# (B,2base,?,?)
        # pad/crop to match skip
        u2 = _match(u2, c2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)          # (B,2base,F,T)
        u1 = F.relu(self.dec1a(u1))# (B,base,?,?)
        u1 = _match(u1, c1)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        logits = self.out(d1)      # (B,2,F,T)
        masks = F.softmax(logits, dim=1)  # sum-to-1 across sources
        return masks


def _match(x, ref):
    """
    Make x spatial dims match ref by center crop or pad.
    """
    _, _, hx, wx = x.shape
    _, _, hr, wr = ref.shape
    # crop
    if hx > hr:
        dh = (hx - hr) // 2
        x = x[:, :, dh:dh + hr, :]
    if wx > wr:
        dw = (wx - wr) // 2
        x = x[:, :, :, dw:dw + wr]
    # pad
    if x.shape[2] < hr:
        pad = hr - x.shape[2]
        x = F.pad(x, (0, 0, pad // 2, pad - pad // 2))
    if x.shape[3] < wr:
        pad = wr - x.shape[3]
        x = F.pad(x, (pad // 2, pad - pad // 2, 0, 0))
    return x
