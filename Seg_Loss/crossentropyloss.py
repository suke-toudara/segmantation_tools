import torch.nn as nn
import torch.nn.functional as F
import torch
class CrossEntropyLoss(nn.Module):
    """
    AuxLoss Param
    ----------
    outputs : Output same size as label
              (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))

    targets : Correct label information
              [num_batch, 475, 475] 

    Returns : loss + auxloss

    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        class_weights = torch.tensor([0.1, 4.0, 1.0,4.0])
        self.Loss = nn.CrossEntropyLoss(reduction='mean') #weight=class_weights,

    def forward(self, outputs, targets):
        loss = self.Loss(outputs, targets)
        return loss