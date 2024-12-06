import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=1.0, eps=1e-9, p: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.p = p

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(self.p).sum(-1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(self.p)
        )
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample

    Choice of p may significantly affect training performance. While original paper proposed p=2, it can lead to vanishing gradients.
    p = 0.5 was later proposed to address this issue
    """

    def __init__(self, margin=1.0, p: float = 2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(self.p).sum(-1)
        distance_negative = (anchor - negative).pow(self.p).sum(-1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
