model=dict(
    type='Pytorch_FasterRCNN',
    params=dict(
        weights=None,
        progress=True,
        weights_backbone=dict(
            type='ResNet50_Weights',
            params=False
        ),
        trainable_backbone_layers=3,
        num_classes=91,
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size_divisible=32,
        fixed_size=None,
        _skip_resize=False,
        rpn_anchor_generator=None,
        # rpn_anchor_generator=dict(
        #     type='AnchorGenerator',
        #     params=dict(
        #         sizes=((32,), (64,), (128,), (256,), (512,)),
        #         aspect_ratios=((0.5, 1.0, 2.0),) * 5
        #     )
        # ),
        rpn_head=None,
        # rpn_head=dict(
        #     type='RPNHead',
        #     params=dict(
        #         in_channels=256,
        #         num_anchors=3,
        #         conv_depth=1
        #     )
        # ),
        # RPN parameters
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        # box_roi_pool=dict(
        #     type='MultiScaleRoIAlign',
        #     params=dict(
        #         featmap_names=["0", "1", "2", "3"], 
        #         output_size=7, 
        #         sampling_ratio=2
        #     )
        # ),
        box_head=None,
        # box_head=dict(
        #     type='TwoMLPHead',
        #     params=dict(
        #         in_channels = 256 * 7 * 7,
        #         representation_size = 1024
        #     )
        # ),
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
    )
)