from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(Downsample, self).__init__()

        kernel_size = scale_factor
        stride = scale_factor

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size, stride, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class ASFF(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class BlockBody(nn.Module):
    def __init__(self, channels):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        self.asff01 = ASFF(inter_dim=channels[0])
        self.asff10 = ASFF(inter_dim=channels[1])
        self.asff23 = ASFF(inter_dim=channels[2])
        self.asff32 = ASFF(inter_dim=channels[3])

        self.upsample1_0 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample3_2 = Upsample(channels[3], channels[2], scale_factor=2)
        self.downsample0_1 = Downsample(channels[0], channels[1], scale_factor=2)
        self.downsample2_3 = Downsample(channels[2], channels[3], scale_factor=2)

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )

        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.blocks_scaletwo2 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.blocks_scalethree2 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

        self.asff02 = ASFF(inter_dim=channels[0])
        self.asff13 = ASFF(inter_dim=channels[1])
        self.asff20 = ASFF(inter_dim=channels[2])
        self.asff31 = ASFF(inter_dim=channels[3])

        self.upsample2_0 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample3_1 = Upsample(channels[3], channels[1], scale_factor=4)
        self.downsample0_2 = Downsample(channels[0], channels[2], scale_factor=4)
        self.downsample1_3 = Downsample(channels[1], channels[3], scale_factor=4)

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )

        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.blocks_scalethree3 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        x01 = self.asff01(x0, self.upsample1_0(x1))
        x10 = self.asff10(self.downsample0_1(x0), x1)
        x23 = self.asff23(x2, self.upsample3_2(x3))
        x32 = self.asff32(self.downsample2_3(x2), x3)

        x0 = self.blocks_scalezero2(x01)
        x1 = self.blocks_scaleone2(x10)
        x2 = self.blocks_scaletwo2(x23)
        x3 = self.blocks_scalethree2(x32)

        x02 = self.asff02(x0, self.upsample2_0(x2))
        x13 = self.asff13(x1, self.upsample3_1(x3))
        x20 = self.asff20(self.downsample0_2(x0), x2)
        x31 = self.asff31(self.downsample1_3(x1), x3)

        x0 = self.blocks_scalezero3(x02)
        x1 = self.blocks_scaleone3(x13)
        x2 = self.blocks_scaletwo3(x20)
        x3 = self.blocks_scalethree3(x31)

        return x0, x1, x2, x3


@MODELS.register_module()
class ABFAN(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor=8):
        super(ABFAN, self).__init__()

        self.reduction_factor = reduction_factor  # Store reduction factor

        # Modify conv layers using reduction_factor
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // reduction_factor, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // reduction_factor, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // reduction_factor, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // reduction_factor, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // reduction_factor,
                       in_channels[1] // reduction_factor,
                       in_channels[2] // reduction_factor,
                       in_channels[3] // reduction_factor])
        )

        self.conv00 = BasicConv(in_channels[0] // reduction_factor, out_channels[0], 1)
        self.conv11 = BasicConv(in_channels[1] // reduction_factor, out_channels[1], 1)
        self.conv22 = BasicConv(in_channels[2] // reduction_factor, out_channels[2], 1)
        self.conv33 = BasicConv(in_channels[3] // reduction_factor, out_channels[3], 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return out0, out1, out2, out3


if __name__ == "__main__":
    batch_size = 1
    input_tensors = [torch.randn(batch_size, 64, 64, 64),
                     torch.randn(batch_size, 128, 32, 32),
                     torch.randn(batch_size, 216, 16, 16),
                     torch.randn(batch_size, 288, 8, 8)]

    bafn = ABFAN([64, 128, 216, 288], out_channels=[32, 64, 128, 256], reduction_factor=4)

    # Forward pass the input through the model
    outputs = bafn(input_tensors)

    # Check the shape of the outputs
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")

