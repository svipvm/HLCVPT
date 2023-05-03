model=dict(
    type='ResRPNNetwork',
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
                        sizes=((128, 256, 512),
                               (128, 256, 512),
                               (128, 256, 512)),
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
                batch_size_per_image=4,
                positive_fraction=0.5,
                pre_nms_top_n=dict(
                    training=1000,
                    testing=1000
                ),
                post_nms_top_n=dict(
                    training=1000,
                    testing=1000
                ),
                nms_thresh=0.3,
                score_thresh=0.0
            )
        )
    )
)