# encoding: utf-8

import os, datetime

def mkdir_if_not_exist(path_list):
    all_path = os.path.join(*path_list)
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    return all_path
