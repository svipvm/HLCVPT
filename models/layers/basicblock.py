# encoding: utf-8

from torch import nn

def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def stack(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for m in mode:
        if m == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif m == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif m == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif m == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif m == 'R':
            L.append(nn.ReLU(inplace=True))
        elif m == 'r':
            L.append(nn.ReLU(inplace=False))
        elif m == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif m == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif m == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif m == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif m == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif m == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif m == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif m == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif m == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif m == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(m))
    return sequential(*L)