_anchor_sizes = (128, 256, 512)
_anchor_aspect_ratios = (0.5, 1.0, 2.0)

model=dict(
    type='ResRPNNetwork',
    params=dict(
        backbone=dict(
            type='ResNet50',
            params=dict(
                pretrained=True
            )
        ),
        rpn=dict(
            type='RegionProposalNetwork',
            params=dict(
                anchor_generator=dict(
                    type='AnchorGenerator',
                    params=dict(
                        sizes=(_anchor_sizes,),
                        aspect_ratios=(_anchor_aspect_ratios,)
                    )
                ),
                head=dict(
                    type='RPNHead',
                    params=dict(
                        in_channels=1024,
                        num_anchors=len(_anchor_sizes)*len(_anchor_aspect_ratios),
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