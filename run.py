import torch
import torch.nn as nn
import random
from collections import defaultdict
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader

from training import train, val, EarlyStopping
from model import Model

def split_data(dataset, train=0.8, val=0.1, seed=42):
    random.seed(seed)
    authors = list(set(dataset['author']))
    random.shuffle(authors)

    n = len(authors)
    train_authors = set(authors[:int(n * train)])
    val_authors = set(authors[int(n * train) : int(n * (train+val))])
    test_authors = set(authors[int(n * (train+val)):])

    return (
        dataset.filter(lambda x: x['author'] in train_authors),
        dataset.filter(lambda x: x['author'] in val_authors),
        dataset.filter(lambda x: x['author'] in test_authors)
    )

class TripleTextDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        super().__init__()

        self.dataset = dataset

        # Make author-indices mapping
        self.labels_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset['author']):
            self.labels_to_indices[label].append(idx)

        self.unique_labels = list(self.labels_to_indices.keys())

    def __len__(self):
        return len(self.dataset)
    
    def _get_embedding(self, idx):
        return torch.tensor(self.dataset[idx]['embedding'], dtype=torch.float)
    
    def _sample_positive(self, idx, label):
        candidates = [i for i in self.labels_to_indices[label] if i != idx]
        return random.choice(candidates)
    
    def _sample_negative(self, label):
        neg_label = random.choice([l for l in self.unique_labels if l != label])
        return random.choice(self.labels_to_indices[neg_label])
    
    def __getitem__(self, idx):
        
        anchor_label = self.dataset[idx]['author']

        pos_idx = self._sample_positive(idx, anchor_label)
        neg_idx = self._sample_negative(anchor_label)

        return {
            'anchor': self._get_embedding(idx),
            'positive': self._get_embedding(pos_idx),
            'negative': self._get_embedding(neg_idx)
            }

if __name__ == "__main__":

    print("Loading processed dataset")
    dataset = load_dataset(path = 'parquet', data_files=['data/blogtext_processed.parquet'], split='train')

    print("Splitting data.. ")
    train_split, val_split, test_split = split_data(dataset)

    print("Building TripleTextDatasets")
    train_ds = TripleTextDataset(train_split)
    val_ds = TripleTextDataset(val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(input_dim=198).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss()
    early_stop = EarlyStopping(patience=10, model_name="model.pt")

    print("Beginning training")
    for epoch in range(1000):

        train_loss = train(model, train_loader, optimiser, criterion, device)
        val_loss = val(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/10] | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f}")

        stop = early_stop.step(model, val_loss, epoch)
        if stop:
            print("Early stopping triggered. Training terminated at epoch {epoch}.")
            break

    print("Training complete")

    