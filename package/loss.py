import torch

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling hard examples.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights
    
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.weights is not None:
             if self.weights.device != inputs.device:
                self.weights = self.weights.to(inputs.device)
             
             # Apply class weights
             weight_per_sample = self.weights[targets]
             focal_loss = focal_loss * weight_per_sample
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss