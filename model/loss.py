#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2020/12/1 22:48
# @Author  : lqyang---jx
# @File    : loss.py
# Desc:
loss 函数类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.Module):
    """
    compute the  loss of model
    """
    def __init__(self, n_negative, beta=1):
        """
        Args:
            log_path_txt: 存输出结果的路径
            n_negative:
            sample_strategy = 'random'  # type=str 'random', 'category', sample strategy for negative samples
            beta: float, default=1.0, balance the negative sample
        """
        super(Loss, self).__init__()
        self.beta = beta
        self.n_negative = n_negative


    def forward(self, result_click, target, neg_sample=None): # fixme 这里把补0的节点也算进去了，后续更改
        """
        Args:
            lamda: torch.scalar, control the impact of info_max_loss
            info_loss: torch.Tensor,  a scalar, the info_max loss
            l2_loss: torch.Tensor,  a scalar, the L2 regularization loss
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size, 1维
            neg_sample: torch.Tensor, batch_size * n_negative, dtype=torch.long

        Returns:
            rs_loss：the whole model loss, torch.Tensor, a scalar
            click_loss: torch.Tensor, a scalar, the click loss
        """

        if self.n_negative == 0:
            #click_loss = self.click_loss_fn(result_click, target)
            click_loss = self.add_neg_loss(result_click, target)
        else:
            click_loss = self.add_neg_loss(result_click, target, neg_sample)


        return click_loss


    def add_neg_loss(self, result_click, target, neg_sample=None):
        """
        Add negative samples into recommendation loss
        Args:
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size, 1维, dtype=torch.long
            neg_sample: torch.Tensor, batch_size * n_negative, dtype=torch.long
        Returns:
            click_loss: torch.Tensor,  a scalar, the click loss
        """
        batch_size, n_items = result_click.shape
        device = result_click.device

        result_click = torch.softmax(result_click, dim=1) # batch_size * n_items
        pos_score = - torch.log(result_click[torch.arange(batch_size), target])  # batch_size

        if neg_sample is None:  # model test
            click_loss = torch.mean(pos_score)
        else:  # model train
            n_negative = neg_sample.shape[1]
            neg_index = torch.arange(0,batch_size).to(device) * n_items
            neg_index = (neg_index.view(-1,1) + neg_sample).flatten()
            neg_score = result_click.take(neg_index).view(-1, n_negative)
            neg_score = - torch.log(1 - neg_score).sum(-1) # batch_size

            click_loss = torch.mean(pos_score + self.beta * neg_score)
        return click_loss


class BPRLoss(nn.Module):
    """
    BPR Loss
    """
    def __init__(self, log_path_txt, sample_stategy):
        super(BPRLoss, self).__init__()

    def forward(self, lamda, info_loss,l2_loss,  result_click, target, neg_sample=None):
        """
        compute BPR loss, reference: the bpr loss in BGCN
        Note: 不同的loss function中均有额外附带 info-loss 的那一部分
        Args:
            l2_loss: torch.Tensor,  a scalar, the L2 regularization loss
            result_click: torch.Tensor， batch_size * n_items
            target: torch.Tensor, batch_size, 1维, dtype=torch.long
            neg_sample: torch.Tensor, batch_size * 1, dtype=torch.long
        Returns:
            click_loss: torch.Tensor,  a scalar, the click loss
        """
        batch_size = result_click.shape[0]

        result_click = torch.softmax(result_click, dim=1)  # batch_size * n_items
        pos_score = result_click[torch.arange(batch_size), target]  # batch_size

        if neg_sample is None:  # model test
            click_loss = torch.mean( - torch.log(pos_score))  # 用的依旧是交叉熵loss 在测试计算click loss 时
        else:# model train
            neg_score = result_click[torch.arange(batch_size), neg_sample.squeeze(-1)]
            # BPR loss
            #click_loss = torch.mean(-torch.log(torch.sigmoid(pos_score - neg_score)) )# batch_size
            click_loss = torch.mean(F.softplus(neg_score - pos_score))

        rs_loss = click_loss + lamda * info_loss + l2_loss

        return rs_loss, click_loss

class SampledSoftmaxLoss(torch.nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self, use_cuda):
        """
        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super(SampledSoftmaxLoss, self).__init__()
        self.loss = torch.nn.Softmax()
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(1)
        target =torch.autograd.Variable(torch.arange(batch_size).long())
        if self.use_cuda:
            target = target.cuda()

        return self.loss(logit, target)


if __name__ == "__main__":
    lamda = 0.2
    info_loss = torch.Tensor([3.0])
    result_click = torch.randn(5,10).sigmoid() #batch_size=5, n_items=10
    target = torch.arange(1,6)
    neg_sample = torch.cat((torch.arange(2,7), torch.arange(3,8))).view(-1,2)
    L = Loss(n_negative=2, beta=1)
    loss = L(lamda, info_loss,result_click, target, neg_sample)
    print(loss)