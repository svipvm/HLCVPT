from .datasets.config_coco import *
from .models.config_faster_rcnn import *
from .runtime.config_default import *

task=dict(
    name='coco-detection',
    version='v1.0',
    devices=[0],
    seed=233
)