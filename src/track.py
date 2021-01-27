from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils.log import logger
from tracking_utils.timer import Timer

from gen_utils import write_results, write_results_custom, \
                write_results_score, get_valid_seq,\
                post_process_cls, get_hist, eval_seq, predict_km, write_video

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

    ### Predict the team labels
    all_hists = predict_km(all_hists)
    all_hists = post_process_cls(all_hists, results)
    ### Write to video
    write_video(dataloader, results, output_video, valid_frames, all_hists, ocr_data, img0)
    ### Write results to a File
    write_results_custom(result_filename, results, all_hists)

    return frame_id, timer.average_time, timer.calls

def eval_seq_jersey(ocr_data, opt, dataloader, result_filename, output_video, save_dir=None, frame_rate=30):
    ### This function is supposed to perform jersey identification for all detected tracks


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


    ### Predict the team labels
    all_hists = predict_km(all_hists)
    all_hists = post_process_cls(all_hists, results)
    ### Write to video
    write_video(dataloader, results, output_video, valid_frames, all_hists, ocr_data, img0)
    ### Write results to a File
    write_results_custom(result_filename, results, all_hists)

    return frame_id, timer.average_time, timer.calls

