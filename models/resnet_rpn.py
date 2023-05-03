import torch
# from torch import nn
import torchvision.models.detection.image_list as image_list
from collections import OrderedDict

class ResRPNNetwork(torch.nn.Module):
    def __init__(self, backbone, neck, rpn):
        super(ResRPNNetwork, self).__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        self.feature_indices=(5, 6, 7)
        self.neck = neck
        self.rpn = rpn

    def forward(self, x, target):
        feature_maps= OrderedDict()

        if self.neck:
            feature_maps["feat_0"] = self.backbone[:self.feature_indices[0]](x)
            for inx in range(1, len(self.feature_indices)):
                feature_maps["feat_{}".format(inx)] = self.backbone[
                        self.feature_indices[inx-1]:self.feature_indices[inx]
                    ](feature_maps["feat_{}".format(inx-1)])
            feature_maps = self.neck(feature_maps)
        else:
            feature_map = self.backbone(x)
            feature_maps["feat_0"] = feature_map

        image_sizes = [x.shape[-2:]] * x.shape[0]
        images = image_list.ImageList(x,image_sizes)
        boxes, losses = self.rpn(images, feature_maps, target)

        return boxes, losses