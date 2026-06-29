from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

from .abfan import ABFAN, ASFF, BasicConv, BasicBlock, Upsample


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), reduction_factor=4, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        self.scales = scales
        self.num_output_channels = num_output_channels

        self.num_scales = len(num_ch_enc)
        self.num_ch_enc = list(num_ch_enc)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.abfan = ABFAN(self.num_ch_enc[1:], self.num_ch_dec[1:], reduction_factor=reduction_factor)

        # Scale zero feature map processing
        self.conv0 = BasicConv(self.num_ch_enc[0], self.num_ch_dec[0], 1)

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(self.num_ch_dec[0], self.num_ch_dec[0], 1),
        )

        self.asff01 = ASFF(inter_dim=self.num_ch_dec[0])

        self.upsample1_0 = Upsample(self.num_ch_dec[1], self.num_ch_dec[0], scale_factor=2)

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(self.num_ch_dec[0], self.num_ch_dec[0]),
            BasicBlock(self.num_ch_dec[0], self.num_ch_dec[0]),
            BasicBlock(self.num_ch_dec[0], self.num_ch_dec[0]),
            BasicBlock(self.num_ch_dec[0], self.num_ch_dec[0]),
        )

        # Define the convolution layers for each scale
        self.convs = OrderedDict()
        for i in range(len(self.num_ch_dec)):
            if i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        # Create a list of the decoder layers
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution for Conv2d and Linear layers."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """Forward pass to generate depth predictions from input features."""
        self.outputs = {}

        x1, x2, x3, x4 = self.abfan(input_features[1:])

        x0 = input_features[0]
        x0 = self.conv0(x0)
        x0 = self.blocks_scalezero1(x0)
        x01 = self.asff01(x0, self.upsample1_0(x1))
        x0 = self.blocks_scalezero2(x01)

        feats = [x0, x1, x2, x3, x4]

        for i in range(self.num_scales):
            x = feats[i]

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs
