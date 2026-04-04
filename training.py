import torch

class EarlyStopping:
    def __init__(self, patience: int, model_name: str, delta: int=1e-4):
        self.patience = patience
        self.best_loss = -float('inf')
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
