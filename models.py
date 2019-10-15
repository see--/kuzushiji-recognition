"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation

"""
from torchvision import models
import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    print(e)
    print("Install timm via: `pip install --upgrade timm`")


def convert_to_inplace_relu(model):
  for m in model.modules():
    if isinstance(m, nn.ReLU):
      m.inplace = True


class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, slug, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number fo input channels
        """

        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048

        else:
            assert False, "Bad slug: %s" % slug

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(num_bottleneck_filters // 2, num_filters)
        self.lateral2 = Conv1x1(num_bottleneck_filters // 4, num_filters)
        self.lateral1 = Conv1x1(num_bottleneck_filters // 8, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward_s4(self, enc0):
        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
            mode="nearest")
        map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
            mode="nearest")
        map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
            mode="nearest")
        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        map1, map2, map3, map4 = self.forward_s4(enc0)
        return enc0, map1, map2, map3, map4


class FPNSegmentation(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, slug, num_classes=4212, num_filters=128,
            num_filters_fpn=256, pretrained=True):
        """Creates an `FPNSegmentation` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid
                       level
          num_filters_fpn: the number of filters in each FPN output pyramid
                           level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number of input channels e.g. 3 for RGB
          output_size. Tuple[int, int] height, width
        """

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(slug=slug, num_filters=num_filters_fpn,
                pretrained=pretrained)
        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))

        self.hm = nn.Conv2d(4 * num_filters, 1, 3, padding=1)

        self.classes_embedding = nn.Sequential(
            nn.Conv2d(4 * num_filters, 4 * num_filters, 3, padding=1),
            nn.ReLU(inplace=True))

        self.classes = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(4 * num_filters, num_classes, 1)
        )

        self.up8 = torch.nn.Upsample(scale_factor=8, mode='nearest')
        self.up4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def gather_embeddings(self, embeddings, centers):
        gathered_embeddings = []
        for sample_index in range(len(centers)):
            center_mask = centers[sample_index, :, 0] != -1
            if center_mask.sum().item() == 0:
                continue
            per_sample_centers = centers[sample_index][center_mask]
            emb = embeddings[sample_index][:, per_sample_centers[:, 1],
                per_sample_centers[:, 0]].transpose(0, 1)
            gathered_embeddings.append(emb)
        gathered_embeddings = torch.cat(gathered_embeddings, 0)

        return gathered_embeddings

    def forward(self, x, centers=None, return_embeddings=False):
        # normalize
        x = x / 127.5 - 1.0
        enc0, map1, map2, map3, map4 = self.fpn(x)

        h4 = self.head4(map4)
        h3 = self.head3(map3)
        h2 = self.head2(map2)
        h1 = self.head1(map1)

        map4 = self.up8(h4)
        map3 = self.up4(h3)
        map2 = self.up2(h2)
        map1 = h1

        final_map = torch.cat([map4, map3, map2, map1], 1)
        hm = self.hm(final_map)
        classes_embedding = self.classes_embedding(final_map)
        if return_embeddings:
            return hm, classes_embedding

        if centers is not None:
            gathered_embeddings = self.gather_embeddings(classes_embedding,
                centers)
            classes = self.classes(gathered_embeddings.unsqueeze(
                -1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        else:
            classes = self.classes(classes_embedding)

        return hm, classes


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = FPNSegmentation('r34')
    X = torch.randn(1, 3, 384, 256)
    hm, classes = model(X)
    print(hm.shape, classes.shape)
