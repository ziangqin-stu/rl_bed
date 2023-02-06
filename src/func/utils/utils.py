import yaml
from configs import *


def load_param(file_path_name):
    with open(file_path_name) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf
