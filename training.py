import torch

def train(model, train_loader, optimiser, criterion):
    model.train()

    total_loss = 0.0
    
    for batch in train_loader:
        optimiser.zero_grad()

        anchor = model(batch['anchor'])
        pos = model(batch['positive'])
        neg = model(batch['negative'])

        loss = criterion(anchor, pos, neg)
        total_loss += loss.item()

        loss.backward()
        optimiser.step()

    return total_loss / len(train_loader)

@torch.no_grad()
def val(model, val_loader, criterion):
    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        anchor = model(batch['anchor'])
        pos = model(batch['positive'])
        neg = model(batch['negative'])
        
        loss = criterion(anchor, pos, neg)
        total_loss += loss.item()

    return total_loss / len(val_loader)
