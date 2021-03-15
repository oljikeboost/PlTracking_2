from mmcv import Config
import _init_paths
from opts import opts
from argparse import Namespace

conf = Config.fromfile('./base_conf.py')

opt = opts().parse()
opt_dict = vars(opt)

for k,v in conf.items():
    if k in opt_dict:
        opt_dict[k] = v
opt = Namespace(**opt_dict)


print(conf)