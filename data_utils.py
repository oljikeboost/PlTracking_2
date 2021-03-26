import os
import json
import glob
import shutil
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from tqdm import tqdm


def get_color(lbl):

    if lbl==0:
        return (0,0,255)
    elif lbl==1:
        return (0,255,0)
    else:
        return None


def post_process_cls(all_cls, all_tracks):
    ### First, we need to get the set of all the tracks
    ### After which, to find its corrsponding classes
    ### And transform/interpolate the classes list
    from collections import Counter
    id_to_cls_list = {}
    for en, (cls, track_id) in enumerate(zip(all_cls, all_tracks)):

        if track_id in id_to_cls_list:
            id_to_cls_list[track_id].append(cls)
        else:
            id_to_cls_list[track_id] = [cls]

    id_to_cls_val = {}
    for track_id, cls_lst in id_to_cls_list.items():
        cls_lst = np.array(cls_lst).flatten().tolist()
        cnt = Counter(cls_lst)
        mst_cmn = cnt.most_common()[0][0]
        id_to_cls_val[track_id] = int(mst_cmn)

    output = []
    for en, track_id in enumerate(all_tracks):
        output.append(id_to_cls_val[track_id])

    return output, id_to_cls_val


def get_all_team_classes(id_dict):
    print("Clustering all teams in progress...")
    anno_dirs = glob.glob('../data/raw_data/*')

    ### Create global dict which maps global player track to its new global team class
    global_id_to_cls_val = {}
    all_cls = list(range(0, 2 * len(anno_dirs)))

    def chunks(l, n):
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    cls_chunks = chunks(all_cls, 2)

    for anno_en, anno_dir in enumerate(tqdm(anno_dirs)):

        ### Process a new game
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
        orig_dir = os.path.join('../../data/playerTrackingFrames', os.path.basename(anno_dir))

        ### Create the corresponding history of labels and histograms
        all_hists = []
        all_labels = []

        anno_error = 0
        box_cnt = 0
        for en, single_json in enumerate(all_jsons):
            data = json.load(open(single_json))

            img_path = os.path.join(orig_dir, os.path.basename(single_json).replace('.json', '.jpg'))
            img0 = cv2.imread(img_path)
            h, w, _ = img0.shape

            for i in range(len(data['shapes'])):
                box_cnt += 1

                label = data['shapes'][i]['label']
                pts = np.array(data['shapes'][i]['points']).astype(int)
                if pts[0][1] > pts[1][1] or pts[0][0] > pts[1][0]:
                    anno_error += 1
                    continue

                player_label = id_dict[os.path.basename(anno_dir)][label]

                center_y = int((pts[1][1] + pts[0][1]) / 2)
                center_x = int((pts[1][0] + pts[0][0]) / 2)

                img_box = img0[max(0, center_y - 30): min(h, center_y + 30),
                          max(0, center_x - 10): min(w, center_x + 10), :]

                img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([img_box], [0], None, [24],
                                    [0, 300])
                hist = cv2.normalize(hist, hist).flatten()

                all_hists.append(hist)
                all_labels.append(player_label)

        concat_hists = np.concatenate(all_hists)
        km = KMeans(n_clusters=2, init="k-means++", max_iter=10000).fit(all_hists)
        proc_cls, id_to_cls_val = post_process_cls(km.labels_, all_labels)

        print(anno_en, anno_dir, Counter(proc_cls), 100 * (anno_error / box_cnt))

        for player_id, color_cls in id_to_cls_val.items():
            curr_cls_subset = cls_chunks[anno_en]
            global_id_to_cls_val[player_id] = curr_cls_subset[color_cls]

    print('Clustering is finished!')
    return proc_cls, global_id_to_cls_val


def get_all_team_classes2(id_dict, anno_dirs):
    print("Clustering all teams in progress...")

    ### Create global dict which maps global player track to its new global team class
    global_id_to_cls_val = {}
    all_cls = list(range(0, 2 * len(anno_dirs)))

    def chunks(l, n):
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    cls_chunks = chunks(all_cls, 2)

    for anno_en, anno_dir in enumerate(tqdm(anno_dirs)):

        ### Process a new game
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
        orig_dir = os.path.join('../../data/playerTrackingFrames2', os.path.basename(anno_dir))
        if not os.path.exists(orig_dir):
            orig_dir = os.path.join('../../data/playerTrackingFrames', os.path.basename(anno_dir))

        ### Create the corresponding history of labels and histograms
        all_hists = []
        all_labels = []

        anno_error = 0
        box_cnt = 0
        for en, single_json in enumerate(all_jsons):
            data = json.load(open(single_json))

            img_path = os.path.join(orig_dir, os.path.basename(single_json).replace('.json', '.jpg'))
            img0 = cv2.imread(img_path)
            h, w, _ = img0.shape

            for i in range(len(data['shapes'])):
                box_cnt += 1

                label = data['shapes'][i]['label']
                if '_' in label: continue

                pts = np.array(data['shapes'][i]['points']).astype(int)
                if pts[0][1] > pts[1][1] or pts[0][0] > pts[1][0]:
                    anno_error += 1
                    continue

                player_label = id_dict[os.path.basename(anno_dir)][label]

                center_y = int((pts[1][1] + pts[0][1]) / 2)
                center_x = int((pts[1][0] + pts[0][0]) / 2)

                img_box = img0[max(0, center_y - 30): min(h, center_y + 30),
                          max(0, center_x - 10): min(w, center_x + 10), :]

                img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([img_box], [0], None, [24],
                                    [0, 300])
                hist = cv2.normalize(hist, hist).flatten()

                all_hists.append(hist)
                all_labels.append(player_label)

        concat_hists = np.concatenate(all_hists)
        km = KMeans(n_clusters=2, init="k-means++", max_iter=10000).fit(all_hists)
        proc_cls, id_to_cls_val = post_process_cls(km.labels_, all_labels)

        #         print(anno_en, anno_dir, Counter(proc_cls), 100 * (anno_error/box_cnt))

        for player_id, color_cls in id_to_cls_val.items():
            curr_cls_subset = cls_chunks[anno_en]
            global_id_to_cls_val[player_id] = curr_cls_subset[color_cls]

    print('Clustering is finished!')
    return proc_cls, global_id_to_cls_val


def get_all_classes(anno_dirs):
    id_dict = {}
    k_class = 1
    for anno_dir in anno_dirs:
        id_dict[os.path.basename(anno_dir)] = {}

        curr_set = set()
        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))
        for single_json in all_jsons:
            data = json.load(open(single_json))

            for i in range(len(data['shapes'])):
                if '_' not in data['shapes'][i]['label']:
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

    return id_dict

def create_mot_third_task(id_dict, id_to_cls_val):
    gt_list = []
    anno_dirs = glob.glob('../data/third_task/*')


    for anno_dir in tqdm(anno_dirs):

        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))


        ### Iterate through all frames of current directory
        cls_en = 0
        gt_list = []
        curr_labels = set()
        for en, single_json in enumerate(all_jsons):
            data = json.load(open(single_json))


            ### The following block of code creates the jersey_dict which maps track_id to [jersey_num, ball_possession]
            jersey_dict = {}                      
            for i in range(len(data['shapes'])):

                label = data['shapes'][i]['label']
                if '_' not in label: continue

                lbl_split = label.split('_')
                if 'j' in label:
                    _, track_id, jersey_num = lbl_split

                    if not track_id in jersey_dict:
                        jersey_dict[str(track_id)] = [jersey_num, 0]
                    else:
                        jersey_dict[str(track_id)][0] = jersey_num

                elif 'b' in label:

                    _, track_id = lbl_split

                    if not track_id in jersey_dict:
                        jersey_dict[track_id] = [None, 1]
                    else:
                        jersey_dict[track_id][1] = 1


            for i in range(len(data['shapes'])):
                bbox = data['shapes'][i]['points']  
                label = data['shapes'][i]['label']
                if '_' in label: continue

                curr_labels.add(label)

                if bbox[0][0] > bbox[1][0] or bbox[0][1] > bbox[1][1]: 
                    continue

                track_label = id_dict[os.path.basename(anno_dir)][label]
                team_lbl = id_to_cls_val[track_label]

                jersey_num, ball_poc = jersey_dict.get(label, [None, 0])

                anno_line = [en+1, track_label, 
                             int(bbox[0][0]), int(bbox[0][1]), 
                             int(bbox[1][0] - bbox[0][0]), int(bbox[1][1] - bbox[0][1]),
                             1, 1, team_lbl, ball_poc]

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

def create_mot_first_second_task(id_dict, id_to_cls_val):
    
    gt_list = []
    anno_dirs = glob.glob('../data/raw_data/*')
    jersey_dir = '../data/second_task/'

    for dr_en, anno_dir in enumerate(tqdm(anno_dirs)):

        jersey_anno = os.path.join(jersey_dir, os.path.basename(anno_dir))

        all_jsons = sorted(glob.glob(anno_dir + '/*.json'))


        ### Iterate through all frames of current directory
        cls_en = 0
        gt_list = []
        curr_labels = set()
        for en, single_json in enumerate(all_jsons):
            data = json.load(open(single_json))

            jersey_file = os.path.join(jersey_anno, os.path.basename(single_json).replace('frame_', ''))

            if os.path.exists(jersey_file):
                jersey_data = json.load(open(jersey_file))   

                ### Map each track for current frame to its existing information, such as Ball Pocession, Jersey Number, Position on Court
                jersey_dict = {}
                for i in range(len(jersey_data['shapes'])):
                    bbox = jersey_data['shapes'][i]['points']  
                    label = jersey_data['shapes'][i]['label']

                    lbl_split = label.split('_')
                    if 'j' in label:

                        _, track_id, jersey_num = lbl_split


                        if not track_id in jersey_dict:
                            jersey_dict[str(track_id)] = [jersey_num, 0]
                        else:
                            jersey_dict[str(track_id)][0] = jersey_num

                    elif 'b' in label:

                        _, track_id = lbl_split

                        if not track_id in jersey_dict:
                            jersey_dict[track_id] = [None, 1]
                        else:
                            jersey_dict[track_id][1] = 1


            for i in range(len(data['shapes'])):
                bbox = data['shapes'][i]['points']  
                label = data['shapes'][i]['label']
                curr_labels.add(label)

                if bbox[0][0] > bbox[1][0] or bbox[0][1] > bbox[1][1]: 
                    continue

                track_label = id_dict[os.path.basename(anno_dir)][label]
                team_lbl = id_to_cls_val[track_label]

                if os.path.exists(jersey_file):
                    jersey_num, ball_poc = jersey_dict.get(label, [None, 0])

                anno_line = [en+1, track_label, 
                             int(bbox[0][0]), int(bbox[0][1]), 
                             int(bbox[1][0] - bbox[0][0]), int(bbox[1][1] - bbox[0][1]),
                             1, 1, team_lbl, ball_poc]

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
 