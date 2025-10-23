import torch, torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        targets = targets.float().unsqueeze(1)
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        loss = self.alpha*(1-pt).pow(self.gamma)*bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

