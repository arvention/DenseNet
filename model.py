import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """
    Dense layer
    """

    def __init__(self,
                 in_channels,
                 expansion_factor=4,
                 growth_rate=32):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bottleneck_size = expansion_factor * growth_rate

        self.conv1x1 = self.get_conv1x1()
        self.conv3x3 = self.get_conv3x3()

    def get_conv1x1(self):
        """
        returns a stack of Batch Normalization, ReLU, and
        1x1 Convolution layers
        """
        layers = []
        layers.append(nn.BatchNorm2d(num_features=self.in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.bottleneck_size,
                                kernel_size=1,
                                stride=1,
                                bias=False))

        return nn.Sequential(*layers)

    def get_conv3x3(self):
        """
        returns a stack of Batch Normalization, ReLU, and
        3x3 Convolutional layers
        """
        layers = []
        layers.append(nn.BatchNorm2d(num_features=self.bottleneck_size))
        layers.append(nn.ReLu(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.bottleneck_size,
                                out_channels=self.growth_rate,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv1x1(x)
        y = self.conv3x3(y)

        y = torch.cat([x, y], 1)

        return y


class DenseBlock(nn.Module):
    """
    Dense block
    """

    def __init__(self,
                 in_channels,
                 num_layers,
                 expansion_factor=4,
                 growth_rate=32):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.growth_rate = growth_rate

        self.net = self.get_network()

    def get_network(self):
        """
        return num_layers dense layers
        """
        layers = []

        for i in range(self.num_layers):
            in_channels = self.in_channels + i * self.growth_rate
            layers.append(DenseBlock(in_channels=in_channels,
                                     expansion_factor=self.expansion_factor,
                                     growth_rate=self.growth_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        return self.layers(x)


class TransitionBlock(nn.Module):
    """
    Transition block
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = self.get_network()

    def get_network(self):
        """
        returns the structure of the block
        """
        layers = []

        layers.append(nn.BatchNorm2d(num_features=self.in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                stride=1,
                                bias=False))
        layers.append(nn.AvgPool2d(kernel_size=2,
                                   stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward pass
        """
        return self.net(x)


"""
different configurations of DenseNet
"""

configs = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '264': [6, 12, 64, 48]
}


class DenseNet(nn.Module):

    """DenseNet Architecture"""

    def __init__(self, config, channels, class_count):
        super(DenseNet, self).__init__()
        self.config = config
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

        self.init_weights()

    def get_conv_net(self):
        """
        returns the convolutional layers of the network
        """
        pass

    def get_fc_net(self):
        """
        returns the fully connected layers of the network
        """
        pass

    def init_weights(self):
        """
        initializes weights for each layer
        """
        pass

    def forward(self, x):
        """
        feed forward
        """
