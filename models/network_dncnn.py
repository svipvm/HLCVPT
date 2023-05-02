# encoding: utf-8

from torch import nn
from .layers import basicblock as B

class DnCNN(nn.Module):
    def __init__(self, in_channels, mod_channels, out_channels, num_layers, act_mode, bias):
        super(DnCNN, self).__init__()

        mod_channel, num_layer = mod_channels[0], num_layers[0]

        m_head = B.stack(in_channels, mod_channel, bias=bias, mode="C"+act_mode[-1])
        m_body = [B.stack(mod_channel, mod_channel, bias=bias, mode="C"+act_mode) for _ in range(num_layer - 2)]
        m_tail = B.stack(mod_channel, out_channels, bias=bias, mode="C")

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        residual = self.model(x)
        return x - residual
