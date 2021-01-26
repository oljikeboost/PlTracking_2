from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import mmcv
import os.path as osp
import cv2
import logging
import argparse
from tqdm import tqdm
import motmetrics as mm
import numpy as np
import torch
from sklearn.cluster import KMeans

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def rand_search():
    params = {}

    params['det_score'] = round(np.random.uniform(0.1, 0.9), 2)
    params['frame_rate'] = np.random.randint(10, 120)
    params['buffer_size'] = np.random.randint(30, 180)
    params['emb_assign'] = round(np.random.uniform(0.1, 0.9), 2)
    params['iou_assign'] = round(np.random.uniform(0.1, 0.9), 2)
    params['unconf_assign'] = round(np.random.uniform(0.1, 0.9), 2)

    return params

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
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


def eval_seq(opt, dataloader, param_dict, result_filename):

    opt.det_thresh = param_dict['det_score']
    frame_rate = param_dict['frame_rate']
    track_buffer_size = param_dict['buffer_size']
    emb_assign = param_dict['emb_assign']
    iou_assign = param_dict['iou_assign']
    unconf_assign = param_dict['unconf_assign']

    tracker = JDETracker(opt, frame_rate,
                         track_buffer_size,
                         emb_assign,
                         iou_assign,
                         unconf_assign

                         )
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):

        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        frame_id += 1

    write_results(result_filename, results, 'mot')
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True, attempt_num=0):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)

    data_type = 'mot'
    param_dict = rand_search()

    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        # logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        nf, ta, tc = eval_seq(opt, dataloader, param_dict, result_filename)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        # logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    metrics = mm.metrics.motchallenge_metrics
    summary = Evaluator.get_summary(accs, seqs, metrics)

    metric_result = {}
    metric_result['idf1'] = summary['idf1']['OVERALL']
    metric_result['mota'] = summary['mota']['OVERALL']
    metric_result['motp'] = summary['motp']['OVERALL']
    metric_result['precision'] = summary['precision']['OVERALL']
    metric_result['recall'] = summary['recall']['OVERALL']

    curr_res = [param_dict, metric_result]
    return curr_res

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if opt.custom_val:
        seqs_str = '''
                    UCLA vs Washington 2-15-20,
                    2020.02.22-Michigan_at_Purdue,
                    2020.02.25-NorthCarolinaState_at_NorthCarolina,
                    2020.02.20-Oregon_at_ArizonaState,
                    2020.02.15-NotreDame_at_Duke
                    '''
        data_root = '/home/ubuntu/oljike/PlayerTracking/data/mot_data/images/train'

    seqs = [seq.strip() for seq in seqs_str.split(',') if seq.strip()!='']

    final_res = {}
    for i in tqdm(range(200)):
        curr_res = main(opt,
                         data_root=data_root,
                         seqs=seqs,
                         exp_name='rand_search',
                         show_image=False,
                         save_images=False,
                         save_videos=False,
                         attempt_num=i)
        final_res[i] = curr_res


        with open('rand_search_results.json', 'w') as f:
            json.dump(final_res, f)

