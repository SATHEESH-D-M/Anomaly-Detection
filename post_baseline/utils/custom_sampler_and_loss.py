import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_num_weights(class_counts, beta=0.9999, device="cpu"):
    """
    Compute class weights using Effective Number of Samples.
    Args:
        class_counts (array-like): counts of each class
        beta (float): hyperparameter, closer to 1 means smoother weighting
    """
    class_counts = np.array(class_counts, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * len(class_counts)  # normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, device="cpu"):
        """
        Args:
            class_counts (list or np.array): number of samples per class
            beta (float): hyperparameter for ENS
            gamma (float): focusing parameter for focal loss
        """
        super().__init__()
        self.gamma = gamma
        self.device = device

        # Compute ENS class weights
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / np.sum(weights) * len(class_counts)  # normalize
        self.class_weights = torch.tensor(
            weights, dtype=torch.float32, device=device
        )

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] with class indices
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", weight=self.class_weights
        )

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze()  # p_t for each sample

        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss
        return loss.mean()
