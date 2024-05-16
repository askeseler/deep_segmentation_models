import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
#resnet = torchvision.models.resnet.resnet18(pretrained=True)
import numpy as np
from SegmentationAutoencoder import *

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet18(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)
        self.metric_fns = {"accuracy": accuracy, "precision": precision, "recall": recall} 

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class UNetWithResnet18Encoder(pl.LightningModule):
    DEPTH = 6

    def __init__(self, n_classes=2, pretrained = True, interpolation="nearest", num_input_channel = 3):
        super().__init__()
        resnet = torchvision.models.resnet.resnet18(pretrained=pretrained)
        if num_input_channel != 3:
            resnet.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)

        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        up_blocks.append(UpBlockForUNetWithResNet18(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet18(in_channels=128 + 128, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet18(in_channels=128, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        self.n_classes = n_classes
        self.valid_dims = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048]
        self.closest_valid_dims = None
        self.dims_before_reshape = None
        self.interpolation = interpolation

    def forward(self, x, with_output_feature_map=False):
        # e.g. x.shape == [1, 3, 273, 409]
        if type(self.closest_valid_dims) == type(None):
            self.dims_before_reshape = [x.shape[2], x.shape[3]]
            self.closest_valid_dims = [self.valid_dims[(np.abs(np.array(self.valid_dims) - x.shape[2])).argmin()],
                                        self.valid_dims[(np.abs(np.array(self.valid_dims) - x.shape[3])).argmin()]]

        #print(x.shape)
        #print([x.shape[0], x.shape[1], self.closest_valid_dims[0],self.closest_valid_dims[1]])
        x = F.interpolate(x, size=self.closest_valid_dims, mode=self.interpolation)

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet18Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet18Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_feature_map = x
        x = self.out(x)
        x = F.interpolate(x, size=self.dims_before_reshape, mode=self.interpolation)

        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        #loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        #loss = F.cross_entropy(y_hat, y)

        self.log("loss/val", loss, on_step=False, on_epoch=True, logger=True, prog_bar = True)

        metrics = {metric: metric_fn(y, y_hat) for metric, metric_fn in self.metric_fns.items()}
        #self.log_dict(metrics, on_step=False, on_epoch=True, logger = True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)

        metrics = {metric: metric_fn(y, y_hat) for metric, metric_fn in self.metric_fns.items()}
        self.log("loss/test", loss, on_step=False, on_epoch=True)
        #self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)