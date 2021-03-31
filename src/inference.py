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
from track import eval_seq_ocr_jersey
from gen_utils import post_process_ocr
logger.setLevel(logging.INFO)

def demo(opt):

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    if os.path.isdir(opt.input_video):
        all_vids = glob.glob(opt.input_video + '/*.mp4')
    else:
        all_vids = [opt.input_video]

    for input_video in all_vids:
        dataloader = datasets.LoadVideo(input_video, opt.img_size)
        basename = os.path.basename(input_video.replace('.mp4', ''))
        result_filename = os.path.join(result_root, basename + '.json')
        output_video_path = osp.join(result_root, basename + '.mp4')

        frame_rate = dataloader.frame_rate
        if opt.ocr is not None:
            ocr_data = open(opt.ocr)
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
