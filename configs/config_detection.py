from .datasets.config_coco import *
from .models.config_rpn_with_fpn import *
from .runtime.config_default import *

task=dict(
    name='coco-detection',
    version='v1.0',
    devices=[2],
    color_type='color',
    seed=233
)