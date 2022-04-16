import torch
import torch.nn as nn
from sklearn import metrics



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class AUCLoss(nn.Module): # custom pytorch loss function that uses rocauc
    def __init__(self,):
        super(AUCLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.topk(input, 1).values.flatten().detach().tolist()
        input = [abs(x) for x in torch.topk(input, 1).values.flatten().detach().tolist()]
        target = target.flatten().tolist()
        auc = metrics.roc_auc_score(target, input)
        auc = torch.tensor(auc, requires_grad=True)
        return auc
