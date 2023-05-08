from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn as Pytorch_FasterRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import torch
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict

class FasterRCNN(torch.nn.Module):
    def __init__(self, transform, backbone, neck, rpn, roi_head, extra_params):
        super(FasterRCNN, self).__init__()
        self.transform = transform
        # select layers that wont be frozen
        backbone_extra_params = extra_params.get('backbone')
        layers_to_train = backbone_extra_params.get('layers_to_train')[
            :backbone_extra_params.get('trainable_layers')
        ]
        # freeze layers only if pretrained backbone is used
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.backbone = IntermediateLayerGetter(backbone, 
                            backbone_extra_params.get('return_layers'))
        self.neck = neck
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, images, targets=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        features = self.neck(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_head(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        return losses if self.training else detections