import torch

def train(model, train_loader, optimiser, criterion, device):
    model.train()

    total_loss = 0.0
    
    for batch in train_loader:
        optimiser.zero_grad()

        anchor = model(batch['anchor'].to(device))
        pos = model(batch['positive'].to(device))
        neg = model(batch['negative'].to(device))

        loss = criterion(anchor, pos, neg)
        total_loss += loss.item()

        loss.backward()
        optimiser.step()

    return total_loss / len(train_loader)

@torch.no_grad()
def val(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        anchor = model(batch['anchor'].to(device))
        pos = model(batch['positive'].to(device))
        neg = model(batch['negative'].to(device))
        
        loss = criterion(anchor, pos, neg)
        total_loss += loss.item()

    return total_loss / len(val_loader)

class EarlyStopping:
    def __init__(self, patience: int, model_name: str, delta: int=1e-4):
        self.patience = patience
        self.best_loss = -float('inf')
        self.best_epoch = 0
        self.count = 0
        self.delta = delta
        self.model_name = model_name

    def step(self, model, loss: float, epoch: int ):
        if loss < self.best_score - self.delta:
            self.best_score = loss
            self.count = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_name)
        else:
            self.count += 1

        return self.count >= self.patience
