# encoding: utf-8
import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *

class TestConfig(unittest.TestCase):

    def test_config(self):
        cfg = load_config('config_detection')

        logger = setup_logger(cfg)
        logger.info('open this config file:\n{}'.format(config_to_str(cfg)))
        
        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
