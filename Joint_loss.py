import sys
import torch
from torch import nn
sys.path.append('.')  # noqa: E402
class FocalLoss(nn.Module):

    def __init__(self, gamma=2.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, target, epoch):
        logp = self.ce(inputs, target)
        P = torch.exp(-logp)
        loss = -torch.mul(torch.pow(torch.sub(1, P), -torch.log(0.1,epoch/100)),logp)#mul是矩阵对应位置相乘，pow是幂运算，mean是求平均，sub减法
        return loss.mean()


class CenterLoss(nn.Module):

    def __init__(self, num_classes=200, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):

        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        print(mask)

        dist = []
        for i in range(batch_size):
            print(mask[i])
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss