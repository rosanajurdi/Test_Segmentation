import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class NoPoolASPP(Module):
    """
    .. image:: _static/img/nopool_aspp_arch.png
        :align: center
        :scale: 25%

    An ASPP-based model without initial pooling layers.

    :param drop_rate: dropout rate.
    :param bn_momentum: batch normalization momentum.

    .. seealso::
        Perone, C. S., et al (2017). Spinal cord gray matter
        segmentation using deep dilated convolutions.
        Nature Scientific Reports link:
        https://www.nature.com/articles/s41598-018-24304-3

    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1,
                 base_num_filters=64):
        super().__init__()

        self.conv1a = nn.Conv2d(1, base_num_filters, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.branch1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=1)
        self.branch1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)
        self.concat_bn = nn.BatchNorm2d(6*base_num_filters, momentum=bn_momentum)

        self.amort = nn.Conv2d(6*base_num_filters, base_num_filters*2, kernel_size=1)
        self.amort_bn = nn.BatchNorm2d(base_num_filters*2, momentum=bn_momentum)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(base_num_filters*2, 1, kernel_size=1)

    def forward(self, x):
        """Model forward pass.

        :param x: input data.
        """
        
        x = F.relu(self.conv1a(x))
        x = self.conv1a_bn(x)
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_bn(x)
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_bn(x)
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_bn(x)
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_bn(branch1)
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_bn(branch1)
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_bn(branch2)
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_bn(branch3)
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_bn(branch3)
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_bn(branch4)
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_bn(branch4)
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_bn(branch5)
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_bn(branch5)
        branch5 = self.branch5b_drop(branch5)

        # Global Average Pooling
        global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1,
                                   branch2,
                                   branch3,
                                   branch4,
                                   branch5,
                                   global_pool], dim=1)

        concatenation = self.concat_bn(concatenation)
        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_bn(amort)
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = F.sigmoid(predictions)

        return predictions


class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x

class BBConv(Module):
    def __init__(self, in_feat, out_feat, pool_ratio):
        super(BBConv, self).__init__()
        self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = F.sigmoid(x)
        return x


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottle neck
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.mp1(x1)

        x2 = self.conv2(p1)
        p2 = self.mp2(x2)

        x3 = self.conv3(p2)
        p3 = self.mp3(x3)

        # Bottom
        x4 = self.conv4(p3)

        # Up-sampling
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        x5 = self.conv9(u3)
        preds = F.softmax(x5)

        return preds


class BB_Unet(Module):
    """A reference U-Net model.
    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1, no_grad=False):
        super(BB_Unet, self).__init__()
        if no_grad is True:
            no_grad_state = True
        else:
            no_grad_state = False
        
        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottle neck
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
        # bounding box encoder path:
        self.b1 = BBConv(1, 256, 4)
        self.b2 = BBConv(1, 128, 2)
        self.b3 = BBConv(1, 64, 1)
        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 4, kernel_size=3, padding=1)

    def forward(self, x, bb):
        x1 = self.conv1(x)
        p1 = self.mp1(x1)

        x2 = self.conv2(p1)
        p2 = self.mp2(x2)

        x3 = self.conv3(p2)
        p3 = self.mp3(x3)

        # Bottle neck
        x4 = self.conv4(p3)
        # bbox encoder
        f1_1 = self.b1(bb)
        f2_1 = self.b2(bb)
        f3_1 = self.b3(bb)
        
        x3_1 = x3*f1_1
        x2_1 = x2*f2_1
        x1_1 = x1*f3_1
        
        # Up-sampling
        u1 = self.up1(x4, x3_1)
        u2 = self.up2(u1, x2_1)
        u3 = self.up3(u2, x1_1)

        x5 = self.conv9(u3)
        preds = F.sigmoid(x5)
        
        return preds


class COnv_Self_Correct_model(Module):
    def __init__(self):
        super(COnv_Self_Correct_model, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x


class Conv_New_Model(Module):
    def __init__(self):
        super(Conv_New_Model, self).__init__()
        self.model_A = BB_Unet()
        self.model_B = Unet()
        self.model_C = COnv_Self_Correct_model()
    
    def forward(self, x, bb, Semi = True):
        if Semi is True:
            with torch.no_grad():
                P_anc = self.model_A(x, bb)
        else:
            P_anc = self.model_A(x, bb)

        P = self.model_B(x)
        y = torch.cat([P_anc, P], dim = 1)
        
        if Semi is True:
            with torch.no_grad():
                y = self.model_C(y)
        else:
            y = self.model_C(y)

        return y

