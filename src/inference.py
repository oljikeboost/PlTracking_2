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
from track import eval_seq_ocr, eval_seq_ocr_jersey
from gen_utils import post_process_ocr
logger.setLevel(logging.INFO)

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    if os.path.isdir(opt.input_video):
        all_vids = glob.glob(opt.input_video + '/*.mp4')
    else:
        opt.ocr = os.path.dirname(opt.ocr)
        all_vids = [opt.input_video]

    jersey = True
    for input_video in all_vids:
        dataloader = datasets.LoadVideo(input_video, opt.img_size)
        basename = os.path.basename(input_video.split('.')[0])
        result_filename = os.path.join(result_root, basename + '.json')
        # if os.path.exists(result_filename):
        #     continue

        frame_rate = dataloader.frame_rate
        if opt.ocr is not None:
            ocr_data = open(os.path.join(opt.ocr, basename + '_ocr.json'))
            ocr_data = json.load(ocr_data)

        ### Post process missing interval
        if jersey:
            ocr_data = post_process_ocr(ocr_data)

        output_video_path = osp.join(result_root, basename + '.mp4')

        if jersey:
            eval_seq_ocr_jersey(ocr_data, opt, dataloader, result_filename, output_video=output_video_path,
                     frame_rate=frame_rate)
        else:
            eval_seq_ocr(ocr_data, opt, dataloader, result_filename, output_video=output_video_path,
                     frame_rate=frame_rate)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    demo(opt)
