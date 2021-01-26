import os
import json
import glob
import shutil
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from IPython.display import Video
from tqdm.notebook import tqdm

anno_dirs = glob.glob('../data/raw_data/*')
id_dict = {}
k_class = 1
for anno_dir in anno_dirs:
    id_dict[os.path.basename(anno_dir)] = {}

    curr_set = set()
    all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
    for single_json in all_jsons:
        data = json.load(open(single_json))

        for i in range(len(data['shapes'])):
            curr_set.add(data['shapes'][i]['label'])

    num_classes = len(curr_set)
    curr_classes = sorted(list(curr_set))

    en = 0
    while en < num_classes:
        id_dict[os.path.basename(anno_dir)][curr_classes[en]] = k_class
        en += 1
        k_class += 1

print("The number of class is ", k_class)
print("The number of dirs is ", len(anno_dirs))