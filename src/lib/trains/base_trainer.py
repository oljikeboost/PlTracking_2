from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat
import torch.nn.functional as F
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class RechekModleWithLoss(torch.nn.Module):
    def __init__(self, model, recheck, loss):
        super(RechekModleWithLoss, self).__init__()
        self.base_model = model
        self.recheck = recheck
        self.loss = loss

    def forward(self, batch):
        prev, curr = batch

        im_blob = prev['input'].cuda()
        with torch.no_grad():
            output, _ = self.base_model(im_blob)  # [-1]
            output = output[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg']
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=10)
            prev_id_feature = _tranpose_and_gather_feat(id_feature, inds)

        im_blob = curr['input'].cuda()
        with torch.no_grad():
            output, orig_feat = self.base_model(im_blob)
            output = output[-1]
            id_feature = output['id']
            curr_raw_feature = F.normalize(id_feature, dim=1)

        procInp = self.recheck.process_raw_inp(curr_raw_feature, prev_id_feature)
        outputs = self.recheck(procInp, orig_feat)

        loss, loss_stats = self.loss(outputs, curr['hm'].cuda())
        return outputs, loss, loss_stats

class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        if not opt.train_recheck:
            self.loss_stats, self.loss = self._get_losses(opt)
            self.model_with_loss = ModleWithLoss(model, self.loss)
        else:
            from models.networks.recheck import Recheck, logisticMSE, LogMSE
            self.recheck = Recheck()
            self.optimizer = torch.optim.Adam(self.recheck.parameters(), opt.lr)
            self.loss_stats = ['loss']
            self.loss = LogMSE()
            self.model_with_loss = RechekModleWithLoss(model, self.recheck, self.loss)

        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            if not opt.train_recheck:
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            if opt.train_recheck:
                batch = batch[1]
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
