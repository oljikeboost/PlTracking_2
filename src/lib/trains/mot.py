from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)

        if opt.color_weight > 0:
            print(opt.color_weight)
            print("Adding color head...")
            self.color_classifier = nn.Linear(self.emb_dim, 82)
            self.s_color = nn.Parameter(-1.05 * torch.ones(1))

        if opt.ball_weight > 0:
            print("Adding ball head...")
            self.ball_classifier = nn.Linear(self.emb_dim, 2)
            self.s_ball = nn.Parameter(-0.5 * torch.ones(1))


        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        color_loss = 0
        ball_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

            if opt.color_weight > 0:
                color_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                color_head = color_head[batch['reg_mask'] > 0].contiguous()
                color_head = self.emb_scale * F.normalize(color_head)
                color_target = batch['colors'][batch['reg_mask'] > 0]

                color_output = self.color_classifier(color_head).contiguous()
                color_loss += self.IDLoss(color_output, color_target)

            if opt.ball_weight > 0:
                ball_head = _tranpose_and_gather_feat(output['ball'], batch['ind'])
                ball_head = ball_head[batch['reg_mask'] > 0].contiguous()
                ball_head = self.emb_scale * F.normalize(ball_head)
                ball_target = batch['ball_poss'][batch['reg_mask'] > 0]

                ball_output = self.ball_classifier(ball_head).contiguous()
                ball_loss += self.IDLoss(ball_output, ball_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)

        if opt.color_weight>0:
            loss += torch.exp(-self.s_color) * color_loss + self.s_color

        if opt.ball_weight > 0:
            loss += torch.exp(-self.s_ball) * ball_loss + self.s_ball


        loss *= 0.5
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class ArcMarginModel(nn.Module):
    def __init__(self, emb_size, num_classes, easy_margin=False, margin_m=0.5, margin_s=64.0):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        ### Loss params
        self.gamma = 2.0
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, label):
        x = F.normalize(input)  ### This part in FairMOT is called id_head and has a size of {40, 128}
        W = F.normalize(self.weight)  ### this is our linear classfier
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        logp = self.ce(output, label)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class MotAngularLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotAngularLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        ### Here we start defining our Angular loss parameters and weights
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        self.arcLossID = ArcMarginModel(self.emb_dim, self.nID)
        self.arcLossColor = ArcMarginModel(self.emb_dim, 82)

        self.s_color = nn.Parameter(-1.05 * torch.ones(1))
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        color_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_loss += self.arcLossID(id_head, id_target)

            if opt.color_weight > 0:
                color_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                color_head = color_head[batch['reg_mask'] > 0].contiguous()
                color_target = batch['colors'][batch['reg_mask'] > 0]

                color_loss += self.arcLossColor(color_head, color_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        if opt.color_weight > 0:
            loss += torch.exp(-self.s_color) * color_loss + self.s_color
        loss *= 0.5
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        if opt.angular_loss:
            loss = MotAngularLoss(opt)
        else:
            loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
