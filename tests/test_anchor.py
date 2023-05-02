# encoding: utf-8
import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from utils.util_data import *

class TestAnchor(unittest.TestCase):

    def test_anchor(self):
        cfg = load_config('config_detection')

        logger = setup_logger(cfg)
        # logger.info('open this config file:\n{}'.format(config_to_str(cfg)))
        result = kmean_anchors()
        # logger = get_current_logger()
        logger.info(result)
        
        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
