import mlconfig
from torch import nn

def nearby_int(x):
    return int(round(x))

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)

class DepthwiseConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1):
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DepthwiseConv, self).__init__(*layers)

@mlconfig.register
class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, shallow=False):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(*self.make_layers(width_mult, shallow))
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(nearby_int(width_mult * 1024), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(width_mult=1.0, shallow=False):
        settings = [
            (32, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
        ]
        if not shallow:
            settings += [(512, 1)] * 5
        settings += [
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_channels = 3
        for i, (filters, stride) in enumerate(settings):
            out_channels = nearby_int(width_mult * filters)
            if i == 0:
                layers += [ConvBNReLU(in_channels, out_channels, stride=stride)]
            else:
                layers += [DepthwiseConv(in_channels, out_channels, stride=stride)]
            in_channels = out_channels
        return layers
