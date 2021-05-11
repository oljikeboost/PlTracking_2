import torch.nn as nn
import torch
import torch.nn.functional as F

class Recheck(nn.Module):

    def __init__(self, ):
        super(Recheck, self).__init__()

        hidden_dim = 256
        inp = 1
        oup = 1
        stride = 1
        self.inverted_bottleneck = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        hidden_dim = 256
        inp = 64
        oup = 1
        self.last_block = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 3, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.Sigmoid()
        )

    def forward(self, x, origFeat):

        invOut = x + self.inverted_bottleneck(x)
        origEnhanced = invOut * origFeat
        return self.last_block(origEnhanced)

    def process_raw_inp(self, raw, proc):

        ### Process previous and current features
        corrFilters = self.correlation(raw, proc)
        sumFilters = self.get_sum_of_filters(corrFilters, radius=2)

        return sumFilters.detach()

    def get_mask(self, corr_filter, radius=4):

        list_inds = []
        max_ind = (corr_filter == torch.max(corr_filter)).nonzero()[0].tolist()
        for i in range(max_ind[0] - radius, max_ind[0] + radius):
            for j in range(max_ind[1] - radius, max_ind[1] + radius):
                if 0<=i<corr_filter.shape[0] and 0<=j<corr_filter.shape[1]:
                    list_inds.append([i, j])

        mask = torch.zeros(corr_filter.shape).cuda()
        for idx in list_inds:
            mask[idx[0], idx[1]] = 1

        return mask

    def correlation(self, x, kernel):
        '''
        feats = (batch_size, emb_size, H, W)
        kernel = (batch_size, N, emb_size)

        output = (batch_size, N, H, W)
        '''
        kernel = kernel.view(kernel.size()[0], kernel.size()[1] * kernel.size()[2])
        kernel = kernel.unsqueeze(-1).unsqueeze(-1)

        batch = x.size()[0]

        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])

        return po

    def get_sum_of_filters(self, corr_filters, radius=3):

        results = []

        for batch in corr_filters:

            final_filt = torch.zeros(batch[0].shape).cuda()
            for filt in batch:
                mask = self.get_mask(filt, radius)
                new_filter = mask * filt

                final_filt += new_filter

            results.append(final_filt.unsqueeze(0))

        return torch.stack(results)

    def restore(self):
        def restore(self, path_to_checkpoint_file):
            self.load_state_dict(torch.load(path_to_checkpoint_file))
            # step = int(path_to_checkpoint_file.split('/')[-1][6:-4])


class LogMSE(torch.nn.Module):
    def __init__(self, ):
        super(LogMSE, self).__init__()

    def forward(self, pred, target):
        loss = torch.mean(
            - target * (1 - pred) * torch.log(pred + 1e-15) - (1 - target) * pred * torch.log(1 - pred + 1e-15))

        loss_stats = {'loss': loss}

        return loss, loss_stats

def logisticMSE(pred, target):
    '''
    pred = (batch, 1, H, W)
    target = (batch, 1, H, W)
    '''

    loss = torch.mean(
        - target * (1 - pred) * torch.log(pred + 1e-15) - (1 - target) * pred * torch.log(1 - pred + 1e-15))

    loss_stats = {'loss': loss}

    return loss, loss_stats