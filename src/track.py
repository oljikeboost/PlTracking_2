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


from .utils import write_results, write_results_custom, write_results_score, get_valid_seq, post_process_cls, get_hist, eval_seq

def eval_seq_ocr(ocr_data, opt, dataloader, result_filename, output_video, save_dir=None, frame_rate=30):

    tracker = JDETracker(opt)
    timer = Timer()
    results = []
    frame_id = 0

    limit = float('inf')
    new_seq = False
    all_hists = []
    valid_frames = set()
    for i, (path, img, img0) in enumerate(dataloader):
        curr_data = ocr_data['results'][str(i)]
        # tracker.update_frame()

        if curr_data['score_bug_present']:

            tracker, new_seq = get_valid_seq(tracker, new_seq, frame_rate, curr_data, ocr_data, i, opt)

            if new_seq:
                valid_frames.add(i)
                # run tracking
                timer.tic()
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                online_targets = tracker.update(blob, img0)
                online_tlwhs = []
                online_ids = []
                online_hists = []
                #online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)

                        hist = get_hist(tlwh, img0)
                        online_hists.append(hist)

                if len(online_hists)==0 :
                    all_hists.append(np.zeros((0,0)))
                else:
                    all_hists.append(np.array(online_hists))
                timer.toc()
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
        else:
            new_seq = False

        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        frame_id += 1
        if i > limit: break

    try:
        concat_hists = np.concatenate([x for x in all_hists if len(x)>0])
    except:
        print("PROBLEMA with ", result_filename)
        return

    km = KMeans(n_clusters=2, init="k-means++", max_iter=1000).fit(concat_hists)
    en = 0
    for i in range(len(all_hists)):
        if len(all_hists[i])==0: continue
        for j in range(len(all_hists[i])):
            all_hists[i][j] = km.labels_[en]
            en += 1

    all_hists = post_process_cls(all_hists, results)

    dataloader.re_init()
    valid = 0
    frame_id = 0

    h, w, _ = img0.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), 60, (w, h))

    for i, (path, img, img0) in enumerate(tqdm(dataloader)):
        if valid >= len(results): break
        curr_data = ocr_data['results'][str(i)]

        # if curr_data['score_bug_present'] and curr_data['game_clock_running']:
        if curr_data['score_bug_present']:

            if i in valid_frames:
                _, online_tlwhs, online_ids, = results[valid]
                cls = all_hists[valid]
                img0 = vis.plot_tracking_team(img0, online_tlwhs, online_ids, classes = cls, frame_id=frame_id - 1,
                                         fps=1. / timer.average_time)
                valid += 1

        out.write(img0)
        frame_id += 1

    out.release()

    write_results_custom(result_filename, results, all_hists)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        # frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        frame_rate = int(60/5)
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if opt.custom_val:

        ### UCLA vs Washington 2-15-20,
        seqs_str = '''
                    2020.02.22-Michigan_at_Purdue,
                    2020.02.25-NorthCarolinaState_at_NorthCarolina,
                    2020.02.20-Oregon_at_ArizonaState,
                    2020.02.15-NotreDame_at_Duke
                    '''
        data_root = '/home/ubuntu/oljike/PlayerTracking/data/mot_data/images/train'

    seqs = [seq.strip() for seq in seqs_str.split(',') if seq.strip()!='']

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.load_model.split('/')[-2] + '_' + str(opt.conf_thres),
         show_image=False,
         save_images=False,
         save_videos=False)
