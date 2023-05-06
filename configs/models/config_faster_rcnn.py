model=dict(
    type='FasterRCNN',
    params=dict(
        transform=dict(
            type='GeneralizedRCNNTransform',
            params=dict(
                min_size=800,
                max_size=1333,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                size_divisible=32,
                fixed_size=None,
            )
        ),
        backbone=dict(
            type='ResNet50',
            params=dict(
                pretrained=True,
                norm_layer=dict(
                    type='FrozenBatchNorm2d',
                    params=False
                )
            ),
            extra_params=dict(
                layers_to_train=['layer4', 'layer3', 'layer2', 'layer1'],
                trainable_layers=3,
                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
            )
        ),
        neck=dict(
            type='FeaturePyramidNetwork',
            params=dict(
                in_channels_list=[256, 512, 1024, 2048],
                out_channels=256,
                extra_blocks=dict(
                    type='LastLevelMaxPool',
                    params=dict()
                ),
                norm_layer=None
            )
        ),
        rpn=dict(
            type='RegionProposalNetwork',
            params=dict(
                anchor_generator=dict(
                    type='AnchorGenerator',
                    params=dict(
                        sizes=((32,), (64,), (128,), (256,), (512,)),
                        aspect_ratios=((0.5, 1.0, 2.0),) * 5
                    )
                ),
                head=dict(
                    type='RPNHead',
                    params=dict(
                        in_channels=256,
                        num_anchors=3,
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
                        featmap_names=["0", "1", "2", "3"], 
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
                        num_classes=91
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