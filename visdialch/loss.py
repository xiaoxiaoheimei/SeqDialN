import torch
import torch.nn as nn
import torch.nn.functional as F 

class NpairLoss(nn.Module):
    '''
    Implement a slightly modified  N-pair loss function proposed in "Improved Deep Metric Learning with Multi-class N-pair Loss Objective"
    Link: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
    '''
    def __init__(self, scale=0.25):    
        super(NpairLoss, self).__init__()
        self.scale = scale

    def forward(self, scores, target):
        '''
        Args: 
            @param: scores (tensor): score of each candidate answer, (batch_size*nb_round, nb_opt)
            @param: target (tensor): index of the ground truth answer, (batch_size*nb_round)
        Return: 
            loss: N-pair loss
        ''' 
        _ , nb_opt = scores.size() 
        indx = target.unsqueeze(-1).expand(-1, nb_opt) #(batch_size*nb_round, nb_opt)
        gt_score = torch.gather(scores, 1, indx) #broadcast the ground truth score along the dimension of candidate answers (batch_size*nb_round, nb_opt)
        scores = (scores - gt_score)/self.scale
        
        #log-sum-exp trick
        score_max = scores.max(-1, keepdim=True)[0] #(batch_size*nb_round, 1)
        loss_grid = score_max + (scores - score_max).exp().sum(-1, keepdim=True).log() #(batch_size*nb_ruond, 1)
        loss = loss_grid.mean() #compute the average loss
        return loss


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GeneralizedCrossEntropyLoss, self).__init__()

    def forward(self, scores, targets):
        return (-targets * F.log_softmax(scores, dim=-1)).sum(-1).mean()
