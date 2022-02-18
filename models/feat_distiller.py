import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatDistiller(nn.Module):
    def __init__(self, 
                 descriptor_dimension,  # feature dimension before projection
                 kernel_size = 1, # conv filter size in the feature projector
                 mode = 'softmax', # softmax works the best
                 out_dim = 64, # feature dimension after projection
                 softargmax_mul=7., # temperature parameter in softmax
                 ):

        super().__init__()
        self.out_dim = out_dim
        self.descriptor_dimension = descriptor_dimension
        self.softargmax_mul = softargmax_mul
        self.reg_conv = nn.Conv2d(
            in_channels=descriptor_dimension,
            out_channels=self.out_dim,
            kernel_size=kernel_size,
            bias=True,
            padding = (kernel_size - 1) // 2,
        )
        self.softplus = nn.Softplus()
        self.mode = mode

    def forward(self, input):
        B, C, H, W = input.shape
        assert self.descriptor_dimension == C
        corr = self.reg_conv(input)
        if self.mode == 'softmax':
            if self.softargmax_mul != 0:
                corr = corr.view(B, self.out_dim, H * W)
                smcorr = F.softmax(self.softargmax_mul * corr, dim=2)
                pred = smcorr.reshape(B, self.out_dim, H, W)
            else:
                pred = corr 
        elif self.mode == 'softplus':
            pred = self.softplus(corr)
        elif self.mode == 'linear':
            pred = corr
        return pred


if __name__ == '__main__':
    desc_dim = 16
    out_dim = 5
    m = FeatDistiller(
        desc_dim,
        kernel_size = 3, 
        mode = 'softmax', 
        out_dim=out_dim,
    )
    x = torch.randn(10, desc_dim, 24, 24)
    with torch.no_grad():
        out = m.forward(x)
    print("output {}".format(out.shape))
