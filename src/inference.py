from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import shutil
import glob
import json
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq_ocr_jersey, eval_seq_ocr_jersey_sequent
from gen_utils import post_process_ocr
logger.setLevel(logging.INFO)

def demo(opt):
    opt.load_model = os.path.join(opt.exp_id, 'model_{}.pth'.format(opt.num_epochs))

    logger.info('Starting tracking...')

    #define if the input video is directory or file:
    # 1. split the opt.input_video by comma
    all_vids = opt.input_video.split(',')
    # 2. check if it is video or directory
    if os.path.isdir(all_vids[0]):


        all_files = glob.glob(all_vids[0] + '/*')
        for file in all_files:
            if os.path.isdir(file):
                result_root = os.path.join(file, 'results')
                mkdir_if_missing(result_root)
                all_vids = glob.glob(file + '/*.mp4')

    else:
        result_root = opt.output_root if opt.output_root != '' else '.'
        mkdir_if_missing(result_root)

    for en, input_video in enumerate(all_vids):
        input_video = input_video.strip()

        dataloader = datasets.LoadVideo(input_video, opt.img_size)
        basename = os.path.basename(input_video.replace('.mp4', ''))
        result_filename = os.path.join(result_root, basename + '.json')
        output_video_path = osp.join(result_root, basename + '.mp4')

        frame_rate = dataloader.frame_rate

        ocr_data = open(input_video.replace('.mp4', '_ocr.json'))
        ocr_data = json.load(ocr_data)

        ### Post process missing intervals in ocr data
        ocr_data = post_process_ocr(ocr_data)
        eval_seq_ocr_jersey(ocr_data, opt, dataloader, result_filename, output_video=output_video_path,
                     frame_rate=frame_rate)

    print("Inference Finished!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    demo(opt)
