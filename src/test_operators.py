from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import torch
from tqdm import tqdm
import glob
import json

from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq_ocr, eval_seq_ocr_jersey, test_clip
from gen_utils import post_process_ocr
logger.setLevel(logging.INFO)
import mmcv
import cv2
from models.model import create_model, load_model
from tracker.jersey_models import JerseyDetector


def cut_and_save(inp_vid, tgt_frame, output_path, rng=300):

    w = inp_vid.width
    h = inp_vid.height

    out_vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), inp_vid.fps, (w, h))

    for frame_num in range(tgt_frame - rng, tgt_frame + int(rng / 2)):
        curr_frame = inp_vid.get_frame(frame_num)
        out_vid.write(curr_frame)

    out_vid.release()


def prepare_clips(input_video, events_data, frame_range=300):

    inp_vid = mmcv.VideoReader(input_video)
    output_dir = os.path.join(os.path.dirname(input_video), 'clips')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tgt_frames = set()
    for en, ev in enumerate(events_data['events']):
        if ev['assist_jersey']:
            tgt_frame = ev['assist_frame_num']
            tgt_frames.add(tgt_frame)
        else:
            continue


        output_path = os.path.join(output_dir, 'clip_' + str(en) + '_' + str(tgt_frame) + '.mp4')
        if os.path.exists(output_path):
            continue
        cut_and_save(inp_vid, tgt_frame, output_path, frame_range)




def demo(opt):

    list_of_test_games = '''
                    2021_01_20_Colorado_at_Washington,
                    2021_01_23_UCLA_at_Stanford,
                    2021_01_31_UNLV_at_Nevada,
                    2021_01_23_VirginiaMilitary_at_Mercer,
                    2021_01_14_Washington_at_USC,
                    2021_01_24_Utah_at_Washington
    '''
    list_of_test_games = [seq.strip() for seq in list_of_test_games.split(',') if seq.strip() != '']

    print("Creating both models")

    opt.load_model = os.path.join('../exp/mot/', opt.exp_id, 'model_{}.pth'.format(opt.num_epochs))
    model = create_model(opt.arch, opt.heads, opt.head_conv, False)
    model = load_model(model, opt.load_model)
    model = model.to(torch.device('cuda'))
    model.eval()

    jersey_detector = JerseyDetector()


    crrt = 0
    all = 0
    input_dir = '/home/ubuntu/oljike/data/videos2/'
    input_games = os.listdir(input_dir)
    input_games = [x for x in input_games if 'ipynb' not in x]
    input_games = [x for x in input_games if x in list_of_test_games]
    for game_en, game in enumerate(tqdm(input_games)):
        game_dir = os.path.join(input_dir, game)

        json_path = glob.glob(game_dir + '/*_processing.json')[0]
        events_data = json.load(open(json_path))

        video_path = glob.glob(game_dir + '/*.mp4')[0]

        frame_range = 300
        prepare_clips(video_path, events_data, frame_range)

        all_clips = glob.glob(os.path.join(game_dir, 'clips') + '/*.mp4')
        all_clips = [x for x in all_clips if 'clip_' in x]

        for en, clip in enumerate(all_clips):
            print("Processing ", clip)
            dataloader = datasets.LoadVideo(clip, opt.img_size)

            events_en = int(os.path.basename(clip).split('_')[1])
            global_num = int(os.path.basename(clip).split('_')[2].replace('.mp4', ''))
            if events_data['events'][events_en]['assist_location'] is None: continue

            if test_clip(model, jersey_detector, events_data['events'][events_en], frame_range, opt, dataloader, global_num):
                crrt += 1
            all += 1

            print("Current accuracy of {} clips is {}".format(all, crrt / all))
        if game_en > 5:
            break

    print("The accuracy of {} clips is {}".format(all, crrt/all))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    demo(opt)
