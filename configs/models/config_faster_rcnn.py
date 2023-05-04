model=dict(
    type='FasterRCNN',
    params=dict(
        backbone=dict(
            type='ResNet50',
            params=dict(
                pretrained=True
            )
        ),
        neck=dict(
            type='FeaturePyramidNetwork',
            params=dict(
                in_channels_list=[256, 512, 1024],
                out_channels=256
            )
        ),
        rpn=dict(
            type='RegionProposalNetwork',
            params=dict(
                anchor_generator=dict(
                    type='AnchorGenerator',
                    params=dict(
                        sizes=((16, 32, 64),
                               (32, 64, 128),
                               (64, 128, 256)),
                        aspect_ratios=((0.5, 1.0, 2.0),
                                       (0.5, 1.0, 2.0),
                                       (0.5, 1.0, 2.0))
                    )
                ),
                head=dict(
                    type='RPNHead',
                    params=dict(
                        in_channels=256,
                        num_anchors=9,
                        conv_depth=1
                    )
                ),
                fg_iou_thresh=0.7,
                bg_iou_thresh=0.3,
                batch_size_per_image=256,
                positive_fraction=0.5,
                pre_nms_top_n=dict(
                    training=2000,
                    testing=1000
                ),
                post_nms_top_n=dict(
                    training=2000,
                    testing=1000
                ),
                nms_thresh=0.7,
                score_thresh=0.0
            )
        ),
        roi_head=dict(
            type='RoIHeads',
            params=dict(
                box_roi_pool=dict(
                    type='MultiScaleRoIAlign',
                    params=dict(
                        featmap_names=["feat_0", "feat_1", "feat_2"], 
                        output_size=7, 
                        sampling_ratio=2
                    )
                ),
                box_head=dict(
                    type='TwoMLPHead',
                    params=dict(
                        in_channels=256 * 7 * 7, 
                        representation_size=1024
                    )
                ),
                box_predictor=dict(
                    type='FastRCNNPredictor',
                    params=dict(
                        in_channels=1024,
                        num_classes=4
                    )
                ),
                # Faster R-CNN training
                fg_iou_thresh=0.5,
                bg_iou_thresh=0.5,
                batch_size_per_image=512,
                positive_fraction=0.25,
                bbox_reg_weights=None,
                # Faster R-CNN inference
                score_thresh=0.05,
                nms_thresh=0.5,
                detections_per_img=100,
            )
        )
    )
)