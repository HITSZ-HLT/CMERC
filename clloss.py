import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def normalization(data):
    for i in range(len(data)):
        _range = torch.max(data[i]) - torch.min(data[i])
        data[i] = (data[i] - torch.min(data[i])) / _range
    return data

class ESCL(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, weight=None, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ESCL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.weight = weight

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if self.weight is not None:
            weight_scl = torch.tensor([self.weight[int(i)] for i in labels]).cuda()
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)      # 16*16
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)

        logist = torch.log(pos / (pos + neg))
        if self.weight is not None:
            loss = -(torch.mean(logist * weight_scl))
        else:
            loss = -(torch.mean(logist))

        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(device)

        return loss

##################################################################################################

class CKSCL(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CKSCL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, weight=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if weight is not None:
            weight_scl = torch.tensor([weight[int(i)] for i in labels]).cuda()
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)      # 16*16
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)

        logist = torch.log(pos / (pos + neg))
        if weight is not None:
            loss = -(torch.mean(logist * weight_scl))
        else:
            loss = -(torch.mean(logist))

        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(device)

        return loss





class CKSCL_base(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CKSCL_base, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, weight=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if weight is not None:
            weight_scl = torch.tensor([weight[int(i)] for i in labels]).cuda()
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)      # 16*16
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)

        logits = torch.log(pos / (pos + neg))
        return logits 


class CKSCL_tav(nn.Module):
    def __init__(self):
        super(CKSCL_tav, self).__init__()
        self.ckscl_base1 = CKSCL_base()
        self.ckscl_base2 = CKSCL_base()
        self.ckscl_base3 = CKSCL_base()

    def forward(self, features, labels1=None, labels2=None, labels3=None, weight1=None, weight2=None, weight3=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        """
        if labels1 != None:
            logits1 = self.ckscl_base1(features, labels1, weight=weight1)
        else:
            logits1 = None
        
        if labels2 != None:
            logits2 = self.ckscl_base2(features, labels2, weight=weight2)
        else:
            logits2 = None
        
        if labels3 != None:
            logits3 = self.ckscl_base3(features, labels3, weight=weight3)
        else:
            logits3 = None

        if logits1 != None and logits2 != None and logits3 != None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1 + torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2 + torch.tensor([weight3[int(i)] for i in labels3]).cuda() * logits3
        elif logits1 != None and logits2 != None and logits3 == None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1 + torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2
        elif logits1 != None and logits2 == None and logits3 != None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1 + torch.tensor([weight3[int(i)] for i in labels3]).cuda() * logits3
        elif logits1 == None and logits2 != None and logits3 != None:
            logits = torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2 + torch.tensor([weight3[int(i)] for i in labels3]).cuda() * logits3
        elif logits1 != None and logits2 == None and logits3 == None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1
        elif logits1 == None and logits2 != None and logits3 == None:
            logits = torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2
        elif logits1 == None and logits2 == None and logits3 != None:
            logits = torch.tensor([weight3[int(i)] for i in labels3]).cuda() * logits3

        loss = -(torch.mean(logits))
        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(features.device)
        return loss





class CKSCL_context(nn.Module):
    def __init__(self):
        super(CKSCL_context, self).__init__()
        self.ckscl_base1 = CKSCL_base()
        self.ckscl_base2 = CKSCL_base()

    def forward(self, features, labels1=None, labels2=None, weight1=None, weight2=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        """
        if labels1 != None:
            logits1 = self.ckscl_base1(features, labels1, weight=weight1)
        else:
            logits1 = None
        
        if labels2 != None:
            logits2 = self.ckscl_base2(features, labels2, weight=weight2)
        else:
            logits2 = None
        
        if logits1 != None and logits2 != None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1 + torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2
        elif logits1 != None and logits2 == None:
            logits = torch.tensor([weight1[int(i)] for i in labels1]).cuda() * logits1
        elif logits1 == None and logits2 != None:
            logits = torch.tensor([weight2[int(i)] for i in labels2]).cuda() * logits2
        elif logits1 == None and logits2 == None:
            logits = None
        
        loss = -(torch.mean(logits))
        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(features.device)
        return loss



