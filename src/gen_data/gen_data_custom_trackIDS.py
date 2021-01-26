import os.path as osp
import os
import numpy as np
import shutil

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)
    else:
        shutil.rmtree(d)
        os.makedirs(d)


seq_root = '/home/ubuntu/oljike/PlayerTracking/data/mot_data/images/train'
label_root = '/home/ubuntu/oljike/PlayerTracking/data/mot_data/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_width = 1280
    seq_height = 720

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, delimiter=',')
    # gt = np.genfromtxt(gt_txt, dtype=np.float64)

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, team_color in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))

        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:d}\n'.format(
            tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, int(team_color))
        with open(label_fpath, 'a') as f:
            f.write(label_str)