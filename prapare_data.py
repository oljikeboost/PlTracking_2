import os
import json
import glob
import shutil
from data_utils import get_all_team_classes
from tqdm.notebook import tqdm

if __name__=="__main__":

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

    gt_list = []
    anno_dirs = glob.glob('../data/raw_data/*')

    _, id_to_cls_val = get_all_team_classes(id_dict)

    for anno_dir in tqdm(anno_dirs):
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))

        gt_list = []
        cls_en = 0
        for en, single_json in enumerate(all_jsons):
            data = json.load(open(single_json))

            for i in range(len(data['shapes'])):
                bbox = data['shapes'][i]['points']
                label = data['shapes'][i]['label']

                if bbox[0][0] > bbox[1][0] or bbox[0][1] > bbox[1][1]:
                    continue

                track_label = id_dict[os.path.basename(anno_dir)][label]
                player_lbl = id_to_cls_val[track_label]

                anno_line = [en + 1, track_label,
                             int(bbox[0][0]), int(bbox[0][1]),
                             int(bbox[1][0] - bbox[0][0]), int(bbox[1][1] - bbox[0][1]),
                             1, 1, player_lbl]

                anno_str = ','.join([str(x) for x in anno_line])

                gt_list.append(anno_str)

        ### Create the output GT dir
        output_dir = os.path.join('../data/mot_data/images/train/', os.path.basename(anno_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'gt')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ### Write the detection to the file gt.txt
        with open(os.path.join(output_dir, 'gt.txt'), 'w') as f:
            for x in gt_list:
                f.writelines(x + '\n')


    anno_dirs = glob.glob('../data/raw_data/*')

    set_of_all = set()
    for anno_dir in anno_dirs:
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
        for js in all_jsons:
            x = '/'.join(js.split('/')[-2:])
            set_of_all.add(x)

    all_dirs = glob.glob('../data/mot_data/images/train/*')
    orig_frames = os.listdir('../../data/playerTrackingFrames/')

    for dr in all_dirs:

        if os.path.basename(dr) in orig_frames:
            orig_dir = os.path.join('../../data/playerTrackingFrames', os.path.basename(dr))

            dest_dir = os.path.join(dr, 'img1')

            if os.path.exists(dest_dir):
                if not os.path.isdir(dest_dir):
                    os.remove(dest_dir)

    all_dirs = glob.glob('../data/mot_data/images/train/*')
    orig_frames = os.listdir('../../data/playerTrackingFrames')

    for dr in all_dirs:

        if os.path.basename(dr) in orig_frames:
            orig_dir = os.path.join('../../data/playerTrackingFrames', os.path.basename(dr))

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

    all_dirs = glob.glob('../data/mot_data/images/train/*')

    for dr in all_dirs:
        img_dr = os.path.join(dr, 'img1')
        curr_imgs = sorted(glob.glob(img_dr + '/*.jpg'))

        for en, img_path in enumerate(curr_imgs):
            base = os.path.basename(img_path)
            new_base = f"{en + 1:06d}.jpg"
            os.rename(img_path, img_path.replace(base, new_base))

    all_dirs = glob.glob('../data/mot_data/images/train/*')
    all_dirs = sorted(all_dirs)

    train_dirs = all_dirs[:int(0.9 * len(all_dirs))]
    val_dirs = all_dirs[int(0.9 * len(all_dirs)):]
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


    with open('src/lib/cfg/custom.json','w') as f:
        json.dump(cfg, f)


