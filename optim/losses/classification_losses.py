import torch
from torch.nn import BCELoss


class CrossEntropyAcrossLines:
    '''
    Average cross entropy loss across all PE lines
    '''

    def __init__(self):
        super(CrossEntropyAcrossLines, self).__init__()
        self.loss_ = BCELoss()

    def __call__(self, target, pred_prob, choose_lines=False, PE_mask=False):
        '''

        :param target:
        :param pred_prob:
        :param choose_lines: optional, if True, then BCE loss is only calculated on lines specified in PE_mask
        :param PE_mask: optional, if choose_lines is True, PE_mask needs to be an array of length target.shape[1],
                        consisting of 0 and 1, depending on whether the corresponding line should be included in BCE loss
        :return:
        '''
        cross_entropy_losses = 0
        for i in range(pred_prob.shape[1]):
            if choose_lines:
                if PE_mask[i] == 1:
                    cross_entropy_losses += self.loss_(pred_prob[:, i].float(), target[:, i].float())
            else:
                cross_entropy_losses += self.loss_(pred_prob[:, i].float(), target[:, i].float())

        return cross_entropy_losses / pred_prob.shape[1]


class WeightedCrossEntropyAcrossLines(torch.nn.Module):

    def __init__(self, weight_cl0=2):
        super(WeightedCrossEntropyAcrossLines, self).__init__()
        self.weight_cl0 = weight_cl0

    def __call__(self, target, pred_prob):
        ce = - (target*torch.log(pred_prob+1e-9) +
                self.weight_cl0 * (1-target)*torch.log((1-pred_prob)+1e-9))

        return torch.mean(ce)
