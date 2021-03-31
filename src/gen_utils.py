from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
from tqdm import tqdm
import os.path as osp
import json
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

import datasets.dataset.jde as datasets
from tracker.multitracker import JDETracker
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer


from tracking_utils.utils import mkdir_if_missing
from opts import opts

ALLOWED = [list(range(0,6)), list(range(10, 16)),
     list(range(20, 26)),list(range(30, 36)),
     list(range(40, 46)),list(range(50, 56)),]

ALLOWED = list(np.array(ALLOWED).flatten())
ALLOWED = [str(x) for x in ALLOWED]
ALLOWED.append('00')
ALLOWED = set(ALLOWED)


def operator_accuracy(events_data, target_num, results, all_jersey, target_frame=None, global_num=None):
    player_location = events_data['assist_location']


    min_dist = float('inf')
    best_jersey = None
    best_location = None
    for curr_res, jersey_list in zip(results, all_jersey):
        frame_id, tlwhs, track_ids = curr_res
        if frame_id != target_num:
            continue

        frame_res = []
        for tlwh, track_id, jersey in zip(tlwhs, track_ids, jersey_list):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h

            bottom_left = np.array((x1, y2))
            bottom_right = np.array((x2, y2))
            bottom_center = np.array(((x1+x2)/2, y2))
            player_location = np.array(player_location)
            # d = np.linalg.norm(np.cross(bottom_right - bottom_left, bottom_left - player_location)) / np.linalg.norm(bottom_right - bottom_left)
            d = np.linalg.norm(bottom_center - player_location)

            if d < (x2 - x1) / 2 and d < min_dist:
                min_dist = d
                best_jersey = jersey
                best_location = tlwh

            # if target_frame is not None:
            #     target_frame = cv2.putText(target_frame, str(int(d)) + '_' + str(jersey), (int((x1+x2)/2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX,
            #                                1, (255, 0, 0), 1)



    tgt_jersey = events_data['assist_jersey']

    crct = False
    if best_jersey is not None and best_jersey==str(tgt_jersey):
        crct = True

    if target_frame is not None:
        if crct:
            clr = (0,255,0)
            txt = 'Match'
            x1, y1, w, h = best_location
            target_frame = cv2.circle(target_frame, (int((x1 + w / 2)), int(y1 + h)), 1, clr, 10, -1)
            target_frame = cv2.putText(target_frame, 'pred_' + best_jersey, (int((x1 + w / 2)) + 15, int(y1 + h)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 1)
        else:
            clr = (0, 0, 255)
            txt = 'Unmatch'

        target_frame = cv2.putText(target_frame, txt, (30,50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 3, clr, 2)


        target_frame = cv2.circle(target_frame, (int(player_location[0]), int(player_location[1])), 1, (255, 0, 0), 10, -1)
        target_frame = cv2.putText(target_frame, 'target_' + str(events_data['assist_jersey']), (int(player_location[0] + 15), int(player_location[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (255, 0, 0), 1)
        cv2.imwrite('dist_results_new/dist_frame_{}_{}.jpg'.format(global_num, crct), target_frame)

    return crct


def write_video(dataloader, results, output_video, valid_frames, all_hists, ocr_data, img0, all_jerseys=None, all_balls=None):

    dataloader.re_init()
    valid = 0
    frame_id = 0

    ### Write to video
    h, w, _ = img0.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), dataloader.frame_rate, (w, h))

    jersey = None
    ball = None
    for i, (path, img, img0) in enumerate(tqdm(dataloader)):
        if valid >= len(results): break
        curr_data = ocr_data['results'][str(i)]

        # if curr_data['score_bug_present'] and curr_data['game_clock_running']:
        if curr_data['score_bug_present']:

            if i in valid_frames:
                _, online_tlwhs, online_ids, = results[valid]
                cls = all_hists[valid]
                if all_jerseys:
                    jersey = all_jerseys[valid]
                if all_balls is not None:
                    ball = all_balls[valid]
                img0 = vis.plot_tracking_team(img0, online_tlwhs, online_ids, classes=cls, jersey=jersey, ball=ball, frame_id=frame_id - 1,
                                              fps=dataloader.frame_rate)
                valid += 1

        out.write(img0)
        frame_id += 1

    out.release()
    # os.system('ffmpeg -y -i {} -vcodec libx264 -loglevel quiet {}'.format(output_video, output_video))


def predict_km(all_hists):

    try:
        concat_hists = np.concatenate([x for x in all_hists if len(x)>0])
    except:
        print("A problem with concatination!")
        return

    km = KMeans(n_clusters=2, init="k-means++", max_iter=1000).fit(concat_hists)
    en = 0
    for i in range(len(all_hists)):
        if len(all_hists[i])==0: continue
        for j in range(len(all_hists[i])):
            all_hists[i][j] = km.labels_[en]
            en += 1

    return all_hists


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_custom(filename, results, classes_list):

    save_format = '{id},{x1},{y1},{w},{h},{cls}'
    save_json = {}

    for curr_res, classes in zip(results, classes_list):
        frame_id, tlwhs, track_ids = curr_res
        frame_res = []
        for tlwh, track_id, cls in zip(tlwhs, track_ids, classes):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            line = save_format.format(id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, cls=cls)
            frame_res.append(line)

        save_json[frame_id] = frame_res

    with open(filename, 'w') as f:
        json.dump(save_json, f)

    logger.info('save results to {}'.format(filename))


def write_results_jersey(filename, results, classes_list, jersey_list, img0):

    imgh, imgw, _ = img0.shape
    save_format = '{id},{x1},{y1},{w},{h},{cls},{jersey}'

    save_json = {}
    for curr_res, classes, curr_jerseys in zip(results, classes_list, jersey_list):
        frame_id, tlwhs, track_ids = curr_res
        frame_res = []
        for tlwh, track_id, cls, jersey in zip(tlwhs, track_ids, classes, curr_jerseys):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h

            line = save_format.format(id=track_id, x1=x1/imgw, y1=y1/imgh, x2=x2/imgw, y2=y2/imgh, w=w, h=h, cls=cls, jersey=jersey)
            frame_res.append(line)

        save_json[frame_id] = frame_res

    with open(filename, 'w') as f:
        json.dump(save_json, f)

    logger.info('save results to {}'.format(filename))


def get_valid_seq(tracker, new_seq, frame_rate, curr_data, ocr_data, i, opt):
    prev_curr = 30
    try:
        curr_time = int(curr_data['regions'][1]['processed_text'].split()[0])
        prev_curr = curr_time
    except:
        curr_time = prev_curr

    prev_fut = curr_time
    if curr_time == 30:
        try:
            future_time = int(ocr_data['results'][str(i + 60)]['regions'][1]['processed_text'].split()[0])
            prev_fut = future_time
        except:
            future_time = prev_fut
    else:
        future_time = None



    if curr_time == 30 and future_time == 29 and not new_seq:
        # tracker = JDETracker(opt, frame_rate=frame_rate)
        tracker.re_init(opt, frame_rate=frame_rate)
        new_seq = True

    elif curr_time == 30 and future_time == 30:
        tracker.re_init(opt, frame_rate=frame_rate)
        new_seq = False

    elif curr_time == 30 and future_time == 29 and new_seq:
        new_seq = True
    elif curr_time <= 29:
        new_seq = True

    return tracker, new_seq


def post_process_ocr(data, thresh=5):
    print("Post processing of ocr missing intervals ...")
    i = 0
    j = 0
    gaps = []
    new_gap = False
    for en in tqdm(range(len(data['results']))):
        if data['results'][str(en)]['score_bug_present'] and data['results'][str(en)]['game_clock_running'] and new_gap:
            gaps.append((j - i, (i, j)))
            new_gap = False
            j += 1
            i = j

        elif data['results'][str(en)]['score_bug_present'] and data['results'][str(en)]['game_clock_running'] and not new_gap:
            j += 1
            i = j

        else:
            j += 1
            new_gap = True

    for gap in gaps:
        if gap[0] <= thresh:
            for idx in range(gap[1][0], gap[1][1]):
                data['results'][str(idx)]['score_bug_present'] = True
                data['results'][str(idx)]['game_clock_running'] = True


    return data


def post_process_cls(all_hists, results, jersey_proc=False):

    ### First, we need to get the set of all the tracks
    ### After which, to find its corrsponding classes
    ### And transform/interpolate the classes list
    from collections import Counter
    id_to_cls_list = {}
    for en, (frame_id, tlwhs, track_ids) in enumerate(results):

        for tlwh, track_id, cls in zip(tlwhs, track_ids, all_hists[en]):
            if track_id in id_to_cls_list:
                id_to_cls_list[track_id].append(cls)
            else:
                id_to_cls_list[track_id] = [cls]

    id_to_cls_val = {}
    for track_id, cls_lst in id_to_cls_list.items():
        cls_lst = np.array(cls_lst).flatten().tolist()
        cnt = Counter(cls_lst)
        mst_cmn = cnt.most_common()
        cmn_1st = mst_cmn[0][0]

        if cmn_1st is None:
            if len(mst_cmn)>1:
                cmn_2nd = mst_cmn[1]
                if cmn_2nd[1]>20 and str(cmn_2nd[0]) in ALLOWED:
                    id_to_cls_val[track_id] = str(cmn_2nd[0])
                else:
                    id_to_cls_val[track_id] = 'None'
            else:
                id_to_cls_val[track_id] = 'None'
        else:
            if jersey_proc and str(cmn_1st) not in ALLOWED:
                id_to_cls_val[track_id] = 'None'
            else:
                id_to_cls_val[track_id] = str(cmn_1st)

    output = []
    for en, (frame_id, tlwhs, track_ids) in enumerate(results):
        curr_output = []
        for j in range(len(track_ids)):
            curr_output.append(id_to_cls_val[track_ids[j]])

        output.append(curr_output)

    return output


def get_hist(tlwh, img0):
    x0, y0, w, h = tlwh
    center_x = int(x0 + w / 2)
    center_y = int(y0 + h / 2)

    # img_large = img[y0:y0+h, x0:x0+w, :]
    img_box = img0[max(0, center_y - 30): min(img0.shape[0], center_y + 30),
              max(0, center_x - 10): min(img0.shape[1], center_x + 10), :]
    if 0 not in img_box.shape:
        img_box = img0[max(0, center_y - 30): min(img0.shape[0], center_y + 30),
                  max(0, center_x - 20): min(img0.shape[1], center_x + 20), :]
    # try:
    #     img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2HSV)
    # except:
    #     print(img_box.shape)
    #     img_box = cv2.cvtColor(prev_box, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('large_{}.jpg'.format(en), img_large)
    # cv2.imwrite('small_{}.jpg'.format(en), img_box)

    hist = cv2.calcHist([img_box], [0], None, [24],
                        [0, 300])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None):

    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = JDETracker(opt)
    timer = Timer()
    results = []
    frame_id = 0
    for i, (path, img, img0) in enumerate(dataloader):
        tracker.update_frame()
        if frame_id % 80 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls
