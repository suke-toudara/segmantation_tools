import torch.nn as nn
import torch.nn.functional as F

class AuxLoss(nn.Module):
    """
    AuxLoss Param
    ----------
    outputs : Output same size as label
              (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))

    targets : Correct label information
              [num_batch, 475, 475] 

    Returns : loss + auxloss

    """
    def __init__(self, aux_weight=0.4):
        super(AuxLoss, self).__init__()
        self.aux_weight = aux_weight  
        self.Loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, outputs, targets):
        loss = self.Loss(outputs[0], targets)
        loss_aux = self.Loss(outputs[1], targets)
        return loss+self.aux_weight*loss_aux
