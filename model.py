import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        # 加载预训练 VGG16
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg16.features.children())

        # 定义五个侧输出对应的特征提取层
        self.dsn1 = nn.Sequential(*features[0:5])   # conv1_2, shape: 64
        self.dsn2 = nn.Sequential(*features[5:10])  # conv2_2, shape: 128
        self.dsn3 = nn.Sequential(*features[10:17]) # conv3_3, shape: 256
        self.dsn4 = nn.Sequential(*features[17:24]) # conv4_3, shape: 512
        self.dsn5 = nn.Sequential(*features[24:31]) # conv5_3, shape: 512

        # 1x1 卷积降维到 1 通道
        self.score_dsn1 = nn.Conv2d(64, 1, kernel_size=1)
        self.score_dsn2 = nn.Conv2d(128, 1, kernel_size=1)
        self.score_dsn3 = nn.Conv2d(256, 1, kernel_size=1)
        self.score_dsn4 = nn.Conv2d(512, 1, kernel_size=1)
        self.score_dsn5 = nn.Conv2d(512, 1, kernel_size=1)

        # 融合层
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        # 只初始化新增的 1x1 卷积层，保持 VGG 预训练权重不变
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.score_dsn1, self.score_dsn2, self.score_dsn3,
                  self.score_dsn4, self.score_dsn5, self.fuse]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]

        # 侧输出 1
        d1 = self.dsn1(x)
        s1 = self.score_dsn1(d1)

        # 侧输出 2
        d2 = self.dsn2(d1)
        s2 = self.score_dsn2(d2)

        # 侧输出 3
        d3 = self.dsn3(d2)
        s3 = self.score_dsn3(d3)

        # 侧输出 4
        d4 = self.dsn4(d3)
        s4 = self.score_dsn4(d4)

        # 侧输出 5
        d5 = self.dsn5(d4)
        s5 = self.score_dsn5(d5)

        # 将所有侧输出上采样到原图尺寸
        s1 = nn.functional.interpolate(s1, size=(img_h, img_w), mode='bilinear', align_corners=False)
        s2 = nn.functional.interpolate(s2, size=(img_h, img_w), mode='bilinear', align_corners=False)
        s3 = nn.functional.interpolate(s3, size=(img_h, img_w), mode='bilinear', align_corners=False)
        s4 = nn.functional.interpolate(s4, size=(img_h, img_w), mode='bilinear', align_corners=False)
        s5 = nn.functional.interpolate(s5, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # 融合
        fuse = torch.cat((s1, s2, s3, s4, s5), dim=1)
        fuse = self.fuse(fuse)

        # Sigmoid 输出概率
        s1 = torch.sigmoid(s1)
        s2 = torch.sigmoid(s2)
        s3 = torch.sigmoid(s3)
        s4 = torch.sigmoid(s4)
        s5 = torch.sigmoid(s5)
        fuse = torch.sigmoid(fuse)

        return s1, s2, s3, s4, s5, fuse