import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    # "flatten" the C * H * W values into a single vector per image
    return x.view(N, -1)


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


class ConvParams:
    def __init__(self, num_filters, filter_size, padding=1, stride=1):
        self.K = num_filters
        self.F = filter_size
        self.p = padding
        self.s = stride


def ConvRelUx2_MaxPool2(conv_params_list, affine_size):
    layers = []
    out_dim = 0
    for i in range(0, len(conv_params_list) - 1, 2):
        if i == 0:
            Kin = 3
            out_dim = 32
        else:
            Kin = conv_params_list[i - 1].K

        K1, K2 = conv_params_list[i].K, conv_params_list[i + 1].K
        F1, F2 = conv_params_list[i].F, conv_params_list[i + 1].F
        p1, p2 = conv_params_list[i].p, conv_params_list[i + 1].p
        s1, s2 = conv_params_list[i].s, conv_params_list[i + 1].s

        layers.extend([nn.BatchNorm2d(Kin)])

        layers.extend([nn.Conv2d(Kin, K1, (F1, F1), padding=p1, stride=s1),
                       nn.BatchNorm2d(K1),
                       nn.ReLU(),
                       nn.Conv2d(K1, K2, (F2, F2), padding=p2, stride=s2),
                       nn.BatchNorm2d(K2),
                       nn.ReLU(),
                       nn.MaxPool2d((2, 2), stride=2)
                       ])

        out_dim = ((out_dim - F1 + 2 * p1) // s1) + 1
        out_dim = ((out_dim - F2 + 2 * p2) // s2) + 1
        out_dim //= 2
    dout = conv_params_list[-1].K * out_dim * out_dim
    layers.extend([Flatten()])
    for i in range(len(affine_size)):
        if i == 0:
            din = conv_params_list[-1].K * out_dim * out_dim
        else:
            din = dout
        dout = affine_size[i]
        layers.extend([nn.Linear(din, dout), nn.BatchNorm1d(dout), nn.ReLU()])

    layers.extend([nn.Linear(dout, 10)])

    return tuple(layers)


class InceptionCellParams:
    def __init__(self, conv1x1_channels, conv3x3r_channels, conv3x3_channels, conv5x5r_channels,
                 conv5x5_channels, max_pool_proj_channels):
        self.conv1x1 = conv1x1_channels
        self.conv3x3r = conv3x3r_channels
        self.conv3x3 = conv3x3_channels
        self.conv5x5r = conv5x5r_channels
        self.conv5x5 = conv5x5_channels
        self.max_pool_proj = max_pool_proj_channels
        self.depth = self.conv1x1 + self.conv3x3 + self.conv5x5 + self.max_pool_proj

    def get(self):
        return self.conv1x1, self.conv3x3r, self.conv3x3, self.conv5x5r, self.conv5x5, self.max_pool_proj


class InceptionCell(nn.Module):
    def __init__(self, in_channel, inception_params):
        super().__init__()

        conv1x1, conv3x3r, conv3x3, conv5x5r, conv5x5, max_pool_proj = inception_params.get()

        self.depth = inception_params.depth

        self.conv1x1 = nn.Conv2d(in_channel, conv1x1, (1, 1),
                                 stride=1, padding=0, bias=True)
        self.bn1x1 = nn.BatchNorm2d(conv1x1)

        self.conv3x3r = nn.Conv2d(in_channel, conv3x3r, (1, 1),
                                  stride=1, padding=0, bias=True)
        self.bn3x3r = nn.BatchNorm2d(conv3x3r)

        self.conv3x3 = nn.Conv2d(conv3x3r, conv3x3, (3, 3),
                                 stride=1, padding=1, bias=True)
        self.bn3x3 = nn.BatchNorm2d(conv3x3)

        self.conv5x5r = nn.Conv2d(in_channel, conv5x5r, (1, 1),
                                  stride=1, padding=0, bias=True)
        self.bn5x5r = nn.BatchNorm2d(conv5x5r)

        self.conv5x5 = nn.Conv2d(conv5x5r, conv5x5, (5, 5),
                                 stride=1, padding=2, bias=True)
        self.bn5x5 = nn.BatchNorm2d(conv5x5)

        self.max_pool = nn.MaxPool2d((3, 3), padding=1, stride=1)

        self.max_pool_proj = nn.Conv2d(in_channel, max_pool_proj, (1, 1),
                                       stride=1, padding=0, bias=True)
        self.bn_max = nn.BatchNorm2d(max_pool_proj)

        nn.init.kaiming_normal_(self.conv1x1.weight)
        nn.init.kaiming_normal_(self.conv3x3r.weight)
        nn.init.kaiming_normal_(self.conv3x3.weight)
        nn.init.kaiming_normal_(self.conv5x5r.weight)
        nn.init.kaiming_normal_(self.conv5x5.weight)
        nn.init.kaiming_normal_(self.max_pool_proj.weight)

    def forward(self, x):
        out1x1 = F.relu(self.conv1x1(x))
        out1x1 = self.bn1x1(out1x1)
        out3x3r = F.relu(self.conv3x3r(x))
        out3x3r = self.bn3x3r(out3x3r)
        out3x3 = F.relu(self.conv3x3(out3x3r))
        out3x3 = self.bn3x3(out3x3)
        out5x5r = F.relu(self.conv5x5r(x))
        out5x5r = self.bn5x5r(out5x5r)
        out5x5 = F.relu(self.conv5x5(out5x5r))
        out5x5 = self.bn5x5(out5x5)
        out_pool = F.relu(self.max_pool_proj(self.max_pool(x)))
        out_pool = self.bn_max(out_pool)
        return torch.cat((out1x1, out3x3, out5x5, out_pool), 1)


class MiniInception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bn_in = nn.BatchNorm2d(3)

        conv1 = ConvParams(64, 5, padding=0)  # 28x28x64
        conv2 = ConvParams(192, 3)  # 28x28x192
        self.conv1 = nn.Conv2d(
            3, conv1.K, (conv1.F, conv1.F), padding=conv1.p, stride=conv1.s)
        self.bn1 = nn.BatchNorm2d(conv1.K)
        self.conv2 = nn.Conv2d(
            conv1.K, conv2.K, (conv2.F, conv2.F), padding=conv2.p, stride=conv2.s)
        self.bn2 = nn.BatchNorm2d(conv2.K)

        self.inception3 = InceptionCell(
            conv2.K, InceptionCellParams(64, 96, 128, 16, 32, 32))  # 28x28x256
        self.inception4 = InceptionCell(
            self.inception3.depth, InceptionCellParams(128, 128, 192, 32, 96, 64))
        # 28x28x480

        self.max_pool5 = nn.MaxPool2d((2, 2), padding=0, stride=2)  # 14x14x480

        conv6 = ConvParams(512, 3)  # 14x14x512
        conv7 = ConvParams(512, 3)  # 14x14x512
        self.conv6 = nn.Conv2d(self.inception4.depth, conv6.K,
                               (conv6.F, conv6.F), padding=conv6.p, stride=conv6.s)
        self.bn6 = nn.BatchNorm2d(conv6.K)
        self.conv7 = nn.Conv2d(
            conv6.K, conv7.K, (conv7.F, conv7.F), padding=conv7.p, stride=conv7.s)
        self.bn7 = nn.BatchNorm2d(conv7.K)

        self.max_pool8 = nn.MaxPool2d((2, 2), padding=0, stride=2)  # 7x7x512

        #         conv9 = ConvParams(256, 1, padding=0)  # 7x7x512
        #         conv10 = ConvParams(256, 1, padding=0)  # 7x7x512
        #         self.conv9 = nn.Conv2d(conv7.K, conv9.K, (conv9.F, conv9.F), padding=conv9.p, stride=conv9.s)
        #         self.conv10 =  nn.Conv2d(conv9.K, conv10.K, (conv10.F, conv10.F), padding=conv10.p, stride=conv10.s)

        self.global_avg_pool = nn.AvgPool2d((7, 7), padding=0, stride=1)

        self.fc = nn.Linear(conv7.K, num_classes)

        #         self.dropout = nn.Dropout(p=0.4)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        #         nn.init.kaiming_normal_(self.conv9.weight)
        #         nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        out = self.bn_in(x)
        out = F.relu(self.conv1(out))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.inception3(out)
        out = self.inception4(out)
        out = self.max_pool5(out)
        out = F.relu(self.conv6(out))
        out = self.bn6(out)
        out = F.relu(self.conv7(out))
        out = self.bn7(out)
        out = self.max_pool8(out)
        #         out = F.relu(self.conv9(out))
        #         out = F.relu(self.conv10(out))

        out = self.global_avg_pool(out)
        out = flatten(out)
        #         out = self.dropout(out)
        scores = self.fc(out)

        return scores
