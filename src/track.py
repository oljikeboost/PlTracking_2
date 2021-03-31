from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from collections import Counter
import numpy as np
import torch


from tracking_utils.log import logger
from tracking_utils.timer import Timer

from gen_utils import write_results, write_results_custom, write_results_jersey, \
                get_valid_seq,post_process_cls, get_hist, eval_seq, predict_km, \
                write_video, operator_accuracy


def get_online_info(tracker, img, img0, opt):

    blob = torch.from_numpy(img).cuda().unsqueeze(0)
    online_targets = tracker.update(blob, img0)
    online_tlwhs = []
    online_ids = []
    online_hists = []
    online_jersey = []
    online_ball = []

    t_jersey, t_ball = None, None
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id

        if hasattr(t, 'jersey_list'):
            t_jersey = t.jersey_list[-1]

        if hasattr(t, 'ball_list'):
            t_ball = t.ball_list[-1]

        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_jersey.append(t_jersey)
            online_ball.append(t_ball)

            hist = get_hist(tlwh, img0)
            online_hists.append(hist)


    return online_tlwhs, online_ids, online_hists, online_jersey, online_ball


def eval_seq_ocr(ocr_data, opt, dataloader, result_filename, output_video, frame_rate=30):

    from tracker.multitracker import JDETracker
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

        if curr_data['score_bug_present']:

            tracker, new_seq = get_valid_seq(tracker, new_seq, frame_rate, curr_data, ocr_data, i, opt)

            if new_seq:
                valid_frames.add(i)
                # run tracking
                timer.tic()
                online_tlwhs, online_ids, online_hists, _, _ = get_online_info(tracker, img, img0, opt)

                if len(online_hists)==0 :
                    all_hists.append(np.zeros((0,0)))
                else:
                    all_hists.append(np.array(online_hists))
                timer.toc()


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
    write_video(dataloader, results, output_video,
                valid_frames, all_hists, ocr_data, img0, None)
    ### Write results to a File
    write_results_custom(result_filename, results, all_hists)

    return frame_id, timer.average_time, timer.calls


def eval_seq_ocr_jersey(ocr_data, opt, dataloader, result_filename, output_video, frame_rate=30):

    if opt.ball_weight>0:
        from tracker.multitracker_jersey_ball import JDETracker
        ball_info = True
        all_balls = []
    else:
        from tracker.multitracker_jersey import JDETracker
        all_balls = None
        ball_info = False

    tracker = JDETracker(opt)
    timer = Timer()
    results = []
    frame_id = 0

    limit = float('inf')
    all_hists = []
    all_jerseys = []
    t_ball = False

    valid_frames = set()
    for i, (path, img, img0) in enumerate(dataloader):
        curr_data = ocr_data['results'][str(i)]

        if curr_data['score_bug_present']:# and curr_data['game_clock_running']:

            valid_frames.add(i)
            timer.tic()

            online_tlwhs, online_ids, online_hists, online_jersey, online_ball = get_online_info(tracker, img, img0, opt)

            if len(online_hists) == 0:
                all_hists.append(np.zeros((0,0)))
            else:
                all_hists.append(np.array(online_hists))

            # save results
            all_jerseys.append(online_jersey)
            if ball_info:
                all_balls.append(online_ball)

            results.append((frame_id + 1, online_tlwhs, online_ids))

            timer.toc()

            if len(valid_frames) > limit: break

        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        frame_id += 1


    ### Predict the team labels
    all_hists = predict_km(all_hists)
    all_hists = post_process_cls(all_hists, results)

    ### Post process for jersey numbers
    all_jerseys = post_process_cls(all_jerseys, results, True)

    ### Post process for ball possession
    #TODO

    ### Write to video
    write_video(dataloader, results, output_video,
                valid_frames, all_hists, ocr_data, img0, all_jerseys, all_balls)

    ### Write results to a File
    write_results_jersey(result_filename, results, all_hists, all_jerseys, img0)

    return frame_id, timer.average_time, timer.calls


def test_clip(model, jerseyDetector, events_data, target_num, opt, dataloader, global_num=None):

    from tracker.multitracker_jersey import JDETracker
    tracker = JDETracker(opt, model=model, jersey_detector=jerseyDetector)
    timer = Timer()
    results = []
    frame_id = 0

    limit = float('inf')
    all_hists = []
    all_jerseys = []

    target_frame = None
    valid_frames = set()
    for i, (path, img, img0) in enumerate(dataloader):

        # we process each frame (without if conditions) because
        # the clips already have only game moments
        if i==target_num:
            target_frame = img0.copy()
        valid_frames.add(i)
        timer.tic()
        online_tlwhs, online_ids, online_hists, online_jersey, _ = get_online_info(tracker, img, img0, opt)

        if len(online_hists)==0 :
            all_hists.append(np.zeros((0,0)))
        else:
            all_hists.append(np.array(online_hists))

        # save results
        all_jerseys.append(online_jersey)
        results.append((frame_id + 1, online_tlwhs, online_ids))

        timer.toc()

        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        frame_id += 1
        if i > limit: break

    ### Predict the team labels
    # all_hists = predict_km(all_hists)
    # all_hists = post_process_cls(all_hists, results)

    ### Post process for jersey numbers
    all_jerseys = post_process_cls(all_jerseys, results, True)

    ### map olayer location from original to new resized
    w_ratio = dataloader.w / dataloader.vw
    h_ratio = dataloader.h / dataloader.vh
    events_data['assist_location'][0] = events_data['assist_location'][0] * w_ratio
    events_data['assist_location'][1] = events_data['assist_location'][1] * h_ratio

    accuracy = operator_accuracy(events_data, target_num, results, all_jerseys, target_frame, global_num)

    return accuracy