import torch
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience: int, model_name: str, delta: int=1e-4):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.count = 0
        self.delta = delta
        self.model_name = model_name

    def step(self, model, loss: float, epoch: int ):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.count = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_name)
        else:
            self.count += 1

        return self.count >= self.patience

def batch_hard_triplet_loss(embeddings, labels, margin=0.2, eps=1e-12):

    # Pairwise Euclidian distance
    dot = torch.mm(embeddings, embeddings.t())
    sq_norm = dot.diagonal()
    dist_sq = (sq_norm.unsqueeze(1) - 2 * dot + sq_norm.unsqueeze(0)).clamp(min=0)
    dist = (dist_sq + eps).sqrt()

    labels = labels.unsqueeze(1) #(B, 1)
    pos_mask = labels.eq(labels.t())
    neg_mask = labels.ne(labels.t())

    # Exclude self from pos mask
    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    pos_mask = pos_mask & ~eye

    # Hardest positive: max distance among positives
    # Replace non-positives with zero so they don't win max
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values

    # Hardest negative: min distance among negatives
    # Replace non-negatives with a large value so they don't win min
    max_dist = dist.max()
    hardest_neg = (dist + (max_dist + 1.0) * (~neg_mask).float()).min(dim=1).values

    # Only calculate loss for anchors that have at least one positive and negative in batch
    valid_anchors = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid_anchors.any():
        return torch.tensor(0.0, requires_grad=True, device=dist.device)
    
    loss = F.relu(hardest_pos[valid_anchors] - hardest_neg[valid_anchors] + margin)
    return loss.mean()