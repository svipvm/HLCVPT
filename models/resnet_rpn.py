import torch
# from torch import nn
import torchvision.models.detection.image_list as image_list

class ResRPNNetwork(torch.nn.Module):
    def __init__(self, backbone, rpn):
        super(ResRPNNetwork, self).__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        self.rpn = rpn

    def forward(self, x, target):
        # print(x.shape)
        feature_map = self.backbone(x)
        # print(feature_map.shape)
        image_sizes = [x.shape[-2:]] * x.shape[0]
        images = image_list.ImageList(x,image_sizes)
        feature_maps= {"0": feature_map}
        boxes, losses = self.rpn(images, feature_maps, target)
        return boxes, losses