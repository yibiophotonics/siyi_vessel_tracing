"""
Discriminative loss function code comes from git hub repository:
https://github.com/nyoki-mtl/pytorch-discriminative-loss/tree/master
Thanks for the author nyoki-mtl for sharing this code.

This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
This implementation is based on following code:
https://github.com/Wizaron/instance-segmentation-pytorch
"""

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch

class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=3, cos_threshold=0.1,
                 norm=2, alpha=1, beta=1, gamma=0.001, epsilon = 0,
                size_average=True, use_cuda=False):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.epsilon = epsilon
        self.cos_threshold = cos_threshold
        self.use_cuda = use_cuda

        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        # _assert_no_grad(target)
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        max_n_clusters = target.size(1)

        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg
        return loss, self.alpha * l_var,self.beta * l_dist, self.gamma * l_reg

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, :n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / target_sample.sum(2)

            # padding
            n_pad_clusters = max_n_clusters - n_clusters[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters)
                pad_sample = Variable(pad_sample)
                if self.use_cuda:
                    pad_sample = pad_sample.cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # print('means',c_means[0])
        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(bs):
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]

            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            # print('var',c_var)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= bs

        return var_term

    def _anisotropic_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = torch.norm((input - c_means), self.norm, 1) * target
        input = input * target

        aniso_term = 0
        for i in range(bs):
            for j in range(n_clusters[i]):
                # n_loc_after_mask
                var_sample = var[i,j]
                # var_sample = var_sample[target[i,j,:] > 0]

                # n_features, n_loc_after_mask
                input_sample = input[i,:,j]
                # input_sample = input_sample[:,target[i,j,:] > 0]

                # n_features, n_loc_after_mask
                c_means_sample = c_means[i,:,j,:]
                # c_means_sample = c_means_sample[:,target[i,j,:] > 0]

                long_axis = torch.max(var_sample, dim=0)
                # n_features
                long_axis_pos = input_sample[:,long_axis.indices]
                # n_features, n_loc
                long_axis_pos = long_axis_pos.unsqueeze(1).expand(n_features, var_sample.size(0))



                # n_loc
                projection = torch.abs(torch.sum((input_sample - c_means_sample)*(long_axis_pos - c_means_sample),dim=0))
                projection = projection / (var_sample * long_axis.values)
                # projection range [0,1]
                possible_short = var_sample[projection <= self.cos_threshold]
                threshold = self.cos_threshold

                punish = 1
                while possible_short.size(0) == 0:
                    threshold += 0.1
                    possible_short = var_sample[projection <= threshold]
                    punish = punish * 2

                # assert possible_short.size()
                # print(possible_short.size())
                short_axis = torch.max(possible_short, dim=0)

                aniso_term +=  -punish * torch.log(short_axis.values / long_axis.values)
            aniso_term /= n_clusters[i]
        aniso_term /= bs
        return aniso_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin)
            if self.use_cuda:
                margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term
