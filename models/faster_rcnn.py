import torch
import torchvision

# Define the Faster R-CNN model
class FasterRCNN(torch.nn.Module):
    def __init__(self, model_cfg):
        super(FasterRCNN, self).__init__()
        # Use a pre-trained ResNet-50 as the backbone
        self.backbone = torchvision.models.resnet50(pretrained=True)
        # Replace the last fully connected layer with a convolutional layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Conv2d(in_features, in_features, kernel_size=1)
        # Define the Region Proposal Network (RPN)
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            in_channels=in_features,
            mid_channels=in_features,
            ratios=[0.5, 1.0, 2.0],
            anchor_sizes=[8, 16, 32],
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7
        )
        # Define the Region of Interest (RoI) Pooling layer
        self.roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        # Replace the last fully connected layer with two sibling layers:
        # a box regression layer and a class prediction layer
        self.box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            in_channels=in_features,
            representation_size=1024
        )
        self.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=1024,
            num_classes=2
        )
        # Define the Faster R-CNN model
        self.faster_rcnn = torchvision.models.detection.FasterRCNN(
            backbone=self.backbone,
            rpn=self.rpn,
            roi_heads=torchvision.models.detection.roi_heads.RoIHeads(
                box_roi_pool=self.roi_pool,
                box_head=self.box_head,
                box_predictor=self.box_predictor
            )
        )

# Create an instance of the Faster R-CNN model
model = FasterRCNN()