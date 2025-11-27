import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_attn = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_attn = nn.BatchNorm2d(out_chan)
        self.sigmoid_attn = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        attn = torch.mean(feat, dim=(2, 3), keepdim=True)
        attn = self.conv_attn(attn)
        attn = self.bn_attn(attn)
        attn = self.sigmoid_attn(attn)
        return feat * attn

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        attn = torch.mean(feat, dim=(2, 3), keepdim=True)
        attn = self.conv1(attn)
        attn = self.bn1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.bn2(attn)
        attn = self.sigmoid(attn)
        return feat + feat * attn

class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256 + 512, 256) # 256 from SP, 512 from CP (128*4? No, let's check CP)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.append(module.weight)
                nowd_params.append(module.bias)
        return wd_params, nowd_params

class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        # ResNet18 forward
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat4 = self.resnet.layer1(x)  # /4
        feat8 = self.resnet.layer2(feat4) # /8
        feat16 = self.resnet.layer3(feat8) # /16
        feat32 = self.resnet.layer4(feat16) # /32

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = F.interpolate(feat32_sum, scale_factor=2, mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, scale_factor=2, mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # feat8 is not used in CP output?
        # Wait, FFM takes feat_sp (128) and feat_cp8?
        # In standard BiSeNet, FFM fuses SP output and CP output (usually 1/8 scale).
        # My implementation of FFM takes 256+512? No.
        # Let's adjust FFM input channels.
        # SP output is 128.
        # CP output here is 128 (feat16_up).
        # So FFM input should be 128 + 128 = 256.

# Re-defining BiSeNet with correct channels
class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        # SP output: 128 channels
        # CP output (feat16_up): 128 channels
        self.ffm = FeatureFusionModule(128 + 128, 256) 
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x) # actually returns feat16_up (1/8 scale relative to original if upsampled? No.)
        # ResNet layer 3 is /16. Up x2 -> /8. So feat16_up is /8 scale.
        # ResNet layer 4 is /32. Up x2 -> /16. So feat32_up is /16 scale.
        
        feat_sp = self.sp(x) # /8 scale
        
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.append(module.weight)
                nowd_params.append(module.bias)
        return wd_params, nowd_params
