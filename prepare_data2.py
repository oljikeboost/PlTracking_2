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
from data_utils import get_all_team_classes2, create_mot_third_task, \
    create_mot_first_second_task, get_all_classes



if __name__=="__main__":



    anno_dirs = glob.glob('../data/third_task/*')
    anno_dirs.extend(glob.glob('../data/raw_data/*'))

    all_dirs = glob.glob('../data/mot_data/images/train/*')

    id_dict = get_all_classes(anno_dirs)
    _, id_to_cls_val = get_all_team_classes2(id_dict, anno_dirs)
    create_mot_third_task(id_dict, id_to_cls_val)
    create_mot_first_second_task(id_dict, id_to_cls_val)


    set_of_all = set()
    for anno_dir in anno_dirs:
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
        for js in all_jsons:
            x = '/'.join(js.split('/')[-2:])
            set_of_all.add(x)


    orig_frames = os.listdir('../../data/playerTrackingFrames')
    orig_frames.extend(os.listdir('../../data/playerTrackingFrames2'))

    for dr in all_dirs:

        if os.path.basename(dr) in orig_frames:
            orig_dir = os.path.join('../../data/playerTrackingFrames', os.path.basename(dr))
            if not os.path.exists(orig_dir):
                orig_dir = os.path.join('../../data/playerTrackingFrames2', os.path.basename(dr))

            dest_dir = os.path.join(dr, 'img1')

            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
                os.makedirs(dest_dir)
            else:
                os.makedirs(dest_dir)

            curr_imgs = glob.glob(orig_dir + '/*.jpg')
            for img in curr_imgs:
                x = '/'.join(img.split('/')[-2:]).replace('.jpg', '.json')
                if x in set_of_all:
                    shutil.copy2(img, dest_dir)


    for dr in all_dirs:
        img_dr = os.path.join(dr, 'img1')
        curr_imgs = sorted(glob.glob(img_dr + '/*.jpg'))

        for en, img_path in enumerate(curr_imgs):
            base = os.path.basename(img_path)
            new_base = f"{en + 1:06d}.jpg"
            os.rename(img_path, img_path.replace(base, new_base))


    seqs_str = '''
                2020.02.22-Michigan_at_Purdue, 
                2020.02.25-NorthCarolinaState_at_NorthCarolina,
                2020.02.20-Oregon_at_ArizonaState, 
                2020.02.15-NotreDame_at_Duke,
                UCLA vs Washington 2-15-20,
                2021_01_20_Colorado_at_Washington,
                2021_01_23_UCLA_at_Stanford, 
                2021_01_31_UNLV_at_Nevada, 
                2021_01_23_VirginiaMilitary_at_Mercer,
                2021_01_14_Washington_at_USC, 
                2021_01_24_Utah_at_Washington
                '''
    data_root = '/home/ubuntu/oljike/PlayerTracking/data/mot_data/images/train'
    val_dirs = [seq.strip() for seq in seqs_str.split(',') if seq.strip() != '']
    val_dirs = [os.path.join('../data/mot_data/images/train/', x) for x in val_dirs]

    train_dirs = [x for x in all_dirs if x not in val_dirs]
    print(len(train_dirs), len(val_dirs))

    output = []
    for dr in train_dirs:
        curr_files = sorted(glob.glob(dr + '/img1/*.jpg'))
        for f in curr_files:
            output.append(f.replace('../data/', ''))

    with open('./src/data/custom.train', 'w') as f:
        for l in output:
            f.writelines(l + '\n')

    print(len(output))

    output = []
    for dr in val_dirs:
        curr_files = sorted(glob.glob(dr + '/img1/*.jpg'))
        for f in curr_files:
            output.append(f.replace('../data/', ''))

    with open('./src/data/custom.val', 'w') as f:
        for l in output:
            f.writelines(l + '\n')

    print(len(output))

    cfg = {}

    cfg['root'] = '/home/ubuntu/oljike/PlayerTracking/data'
    cfg['train'] = {}
    cfg['train']['custom'] = './data/custom.train'
    cfg['test'] = {}
    cfg['test']['custom'] = './data/custom.val'
    cfg['test_emb'] = './data/custom.val'

    with open('src/lib/cfg/custom.json', 'w') as f:
        json.dump(cfg, f)

    print("We are done!")