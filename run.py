import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from data import split_data, AuthorshipDataset, BalancedBatchSampler
from loss import batch_hard_triplet_loss
from training import EarlyStopping
from model import Model
    
if __name__ == "__main__":

    print("Loading processed dataset")
    dataset = load_dataset(path = 'parquet', data_files=['data/blogtext_processed.parquet'], split='train')

    print("Splitting data.. ")
    train_split, val_split, test_split = split_data(dataset)

    print("Building datasets")
    train_ds = AuthorshipDataset(train_split)
    val_ds = AuthorshipDataset(val_split)

    print("Building samplers")
    P = 32
    K = 8
    train_sampler = BalancedBatchSampler(train_ds.dataset['author'], P=P, K=K)
    val_sampler   = BalancedBatchSampler(val_ds.dataset['author'],   P=P, K=K)

    print("Building dataloaders")
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(input_dim=198).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    early_stop = EarlyStopping(patience=10, model_name="model.pt")

    print("Beginning training", flush=True)
    for epoch in range(1000):

        model.train()
        total_train_loss = 0.0
        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimiser.zero_grad()
            outputs = model(embeddings_batch)
            outputs = F.normalize(outputs, p=2, dim=1)

            loss = batch_hard_triplet_loss(outputs, labels_batch)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for embeddings_batch, labels_batch in val_loader:
                embeddings_batch = embeddings_batch.to(device)
                labels_batch = labels_batch.to(device)
 
                outputs = model(embeddings_batch)
                outputs = F.normalize(outputs, p=2, dim=1)
                loss = batch_hard_triplet_loss(outputs, labels_batch)
                total_val_loss += loss.item()
 
        val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/1000] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}",
            flush=True,
        )

        if early_stop.step(model, val_loss, epoch):
            print(f"Early stopping triggered. Training terminated at epoch {epoch + 1}.")
            break

    print("Training complete!")