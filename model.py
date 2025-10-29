import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1,
                      2)  # 将维度顺序从 [batch_size, height, width, channels] 调整为 [batch_size, channels, height, width]

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)  # 使用 .reshape() 方法替代 .view() 方法
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNNBiLSTM(nn.Module):
    def __init__(self):
        super(CNNBiLSTM, self).__init__()
        # CNN部分
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # BiLSTM部分
        self.lstm = nn.LSTM(input_size=128 * 25 * 25, hidden_size=64, num_layers=2, batch_first=True,
                            bidirectional=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # CNN特征提取
        x = x.permute(0, 3, 1,
                      2)  # 将维度顺序从 [batch_size, height, width, channels] 调整为 [batch_size, channels, height, width]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # 将特征展平以输入BiLSTM
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # 使用 .reshape() 方法替代 .view() 方法，展平特征

        # BiLSTM序列建模
        x = x.unsqueeze(1)  # 添加一个维度作为时间步长
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层
        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
def ResNet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    # 加载预训练模型
    model.load_state_dict(torch.load("D:/deeplearning/TAIFENG/resnet18.pth"),strict=False)

    # pretrained_model = models.resnet18(pretrained=True)
    # # 获取预训练模型的状态字典
    # pretrained_dict = torch.load("resnet18.pth")
    # # # 获取当前模型的状态字典
    # model_dict = model.state_dict()
    # # # 删除预训练模型中不匹配的权重
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # # # 更新当前模型的状态字典
    # model_dict.update(pretrained_dict)
    # 加载更新后的状态字典
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
    model.fc = nn.Linear(model.fc.in_features, 1)  # 修改最后的全连接层

    return model

# 定义ResNet-18模型
# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])
#
#
# # 示例用法
# if __name__ == '__main__':
#     # 模型初始化
#     model = ResNet18(num_classes=10)  # 替换为你的分类数量
#
#     # 示例输入
#     sample_input = torch.randn(1, 3, 224, 224)  # 适应 ResNet-18 的输入尺寸
#
#     # 前向传播
#     output = model(sample_input)
#     print("Output shape:", output.shape)
class ResNetRegressor(nn.Module):
    def __init__(self):
        super(ResNetRegressor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        return self.resnet(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 这里修正

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class ResNetRegressorCBAM(nn.Module):
    def __init__(self):
        super(ResNetRegressorCBAM, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        # self.ca1 = ChannelAttention(512)
        # self.sa1 = SpatialAttention()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.cbam1(x)  # 添加CBAM模块
        x = self.resnet.layer2(x)
        x = self.cbam2(x)  # 添加CBAM模块
        x = self.resnet.layer3(x)
        x = self.cbam3(x)  # 添加CBAM模块
        x = self.resnet.layer4(x)
        x = self.cbam4(x)  # 添加CBAM模块
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ResNetRegressorCA(nn.Module):
    def __init__(self):
        super(ResNetRegressorCA, self).__init__()
        self.resnet = models.resnet18(pretrained=True, progress=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        # self.CA1 = CoordAtt(64,64)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.CA1 = CoordAtt(64, 64)
        self.CA2 = CoordAtt(128, 128)
        self.CA3 = CoordAtt(256, 256)
        self.CA4 = CoordAtt(512, 512)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.CA4 = CoordAtt(512,512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.CA1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.CA1(x)  # 添加CA模块
        x = self.resnet.layer2(x)
        x = self.CA2(x)  # 添加CA模块
        x = self.resnet.layer3(x)
        x = self.CA3(x)  # 添加CA模块
        x = self.resnet.layer4(x)
        x = self.CA4(x)  # 添加CA模块

        # x = self.CA4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetRegressorSE(nn.Module):
    def __init__(self):
        super(ResNetRegressorSE, self).__init__()
        self.resnet = models.resnet18(pretrained=True, progress=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        # self.CA1 = CoordAtt(64, 64)
        # self.CA2 = CoordAtt(128, 128)
        # self.CA3 = CoordAtt(256, 256)
        # self.CA4 = CoordAtt(512, 512)
        self.SE = SEModel(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.CA4 = CoordAtt(512,512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # x = self.SE(x)
        x = self.resnet.layer1(x)
        # x = self.CA1(x)  # 添加CA模块
        x = self.resnet.layer2(x)
        # x = self.CA2(x)  # 添加CA模块
        x = self.resnet.layer3(x)
        # x = self.CA3(x)  # 添加CA模块
        x = self.resnet.layer4(x)
        # x = self.CA4(x)  # 添加CA模块

        # x = self.CA4(x)
        x = self.SE(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class ResNetRegressorCBAMSE(nn.Module):
    def __init__(self):
        super(ResNetRegressorCBAMSE, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.SE = SEModel(512)
        # self.ca1 = ChannelAttention(512)
        # self.sa1 = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.cbam1(x)  # 添加CBAM模块
        x = self.resnet.layer2(x)
        x = self.cbam2(x)  # 添加CBAM模块
        x = self.resnet.layer3(x)
        x = self.cbam3(x)  # 添加CBAM模块
        x = self.resnet.layer4(x)
        x = self.cbam4(x)  # 添加CBAM模块
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = self.SE(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class ResNetRegressorCASE(nn.Module):
    def __init__(self):
        super(ResNetRegressorCASE, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        self.CA1 = CoordAtt(64, 64)
        self.CA2 = CoordAtt(128, 128)
        self.CA3 = CoordAtt(256, 256)
        self.CA4 = CoordAtt(512, 512)
        self.SE = SEModel(512)
        # self.ca1 = ChannelAttention(512)
        # self.sa1 = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.CA1(x)
        x = self.resnet.layer2(x)
        x = self.CA2(x)
        x = self.resnet.layer3(x)
        x = self.CA3(x)
        x = self.resnet.layer4(x)
        x = self.CA4(x)
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = self.SE(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1,bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.dilation = dilation
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(in_channels, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class ResNetRegressorDCN(nn.Module):
    def __init__(self):
        super(ResNetRegressorDCN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = DeformConv2d(3, 64, kernel_size=7, padding=3, stride=2, dilation=1, bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # x = self.SE(x)
        x = self.resnet.layer1(x)
        # x = self.CA1(x)  # 添加CA模块
        x = self.resnet.layer2(x)
        # x = self.CA2(x)  # 添加CA模块
        x = self.resnet.layer3(x)
        # x = self.CA3(x)  # 添加CA模块
        x = self.resnet.layer4(x)
        # x = self.CA4(x)  # 添加CA模块

        # x = self.CA4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


class ResNetRegressorDCNSE(nn.Module):
    def __init__(self):
        super(ResNetRegressorDCNSE, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = DeformConv2d(3, 64, kernel_size=7, padding=3, stride=2, dilation=1, bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
        self.SE1 = SEModel(64)
        self.SE2 = SEModel(128)
        self.SE3 = SEModel(256)
        self.SE4 = SEModel(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5)  # 加入Dropout层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层
        # self.SE2 = SEModel(512)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # x = self.SE1(x)
        x = self.resnet.layer1(x)
        x = self.SE1(x)
        x = self.resnet.layer2(x)
        x = self.SE2(x)
        x = self.resnet.layer3(x)
        x = self.SE3(x)
        x = self.resnet.layer4(x)
        # x = self.CA4(x)
        # x = self.CA4(x)
        x = self.SE4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)  # 应用Dropout
        x = self.resnet.fc(x)
        return x


class ResNetRegressorDCNCBAMSE(nn.Module):
    def __init__(self):
        super(ResNetRegressorDCNCBAMSE, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.conv1 = DeformConv2d(3, 64, kernel_size=7, padding=3, stride=2, dilation=1, bias=False)  # 修改第一层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        # self.conv2 = DeformConv2d(64, 128, kernel_size=7, padding=3, stride=2, dilation=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)  # 新增的卷积层，用于将通道数转换回64
        # self.bn3 = nn.BatchNorm2d(64)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.SE = SEModel(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.resnet.layer1(x)
        x = self.cbam1(x)  # 添加CBAM模块
        x = self.resnet.layer2(x)
        x = self.cbam2(x)  # 添加CBAM模块
        x = self.resnet.layer3(x)
        x = self.cbam3(x)  # 添加CBAM模块
        x = self.resnet.layer4(x)
        x = self.cbam4(x)  # 添加CBAM模块

        x = self.SE(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                       nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                       nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                       nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)  # 将输出的特征与原始输入特征结合


class ResNetRegressorSP(nn.Module):
    def __init__(self):
        super(ResNetRegressorSP, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)

        self.strip_pool4 = StripPooling(512, up_kwargs={'mode': 'bilinear', 'align_corners': True})

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)

        x = self.resnet.layer3(x)

        x = self.resnet.layer4(x)
        x = self.strip_pool4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class ResNetRegressorDCNSESP(nn.Module):
    def __init__(self):
        super(ResNetRegressorDCNSESP, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # 修改第一层卷积层为Deformable Convolution
        self.resnet.conv1 = DeformConv2d(3, 64, kernel_size=7, padding=3, stride=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)

        # 为每个ResNet层添加SE模块
        self.se1 = SEModel(64)
        self.se2 = SEModel(128)
        self.se3 = SEModel(256)
        self.se4 = SEModel(512)

        self.strip_pool4 = StripPooling(512, up_kwargs={'mode': 'bilinear', 'align_corners': True})

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 修改最后的全连接层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.se1(x)  # 在layer1之后应用SE模块

        x = self.resnet.layer2(x)
        x = self.se2(x)  # 在layer2之后应用SE模块

        x = self.resnet.layer3(x)
        x = self.se3(x)  # 在layer3之后应用SE模块

        x = self.resnet.layer4(x)
        x = self.se4(x)  # 在layer4之后应用SE模块

        x = self.strip_pool4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
