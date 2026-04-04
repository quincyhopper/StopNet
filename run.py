import torch
import torch.nn.functional as F
import random
from collections import defaultdict
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader, Sampler

from training import EarlyStopping
from model import Model

def split_data(dataset, train_size=0.8, val_size=0.1, seed=42):
    random.seed(seed)
    authors = list(set(dataset['author']))
    random.shuffle(authors)

    n = len(authors)
    train_authors = set(authors[:int(n * train_size)])
    val_authors = set(authors[int(n * train_size) : int(n * (train_size+val_size))])
    test_authors = set(authors[int(n * (train_size+val_size)):])

    return (
        dataset.filter(lambda x: x['author'] in train_authors),
        dataset.filter(lambda x: x['author'] in val_authors),
        dataset.filter(lambda x: x['author'] in test_authors)
        )

class AuthorshipDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        super().__init__()

        self.dataset = dataset.with_format(None)

        unique_authors = self.dataset.unique('author')
        self.label_to_idx = {label: i for i, label in enumerate(sorted(unique_authors))}

    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, idx):
        row = self.dataset[idx]
        return (
            torch.tensor(row['embedding'], dtype=torch.float),
            self.label_to_idx[row['author']]
        )
    
class BalancedBatchSampler(Sampler):
 
    def __init__(self, labels: list, P: int, K: int):
        """
        Args:
            labels: list of author labels, one per dataset index
            P:      number of distinct authors per batch
            K:      number of samples per author per batch
        """
        super().__init__()
        self.P = P
        self.K = K
 
        # Build author-indices mapping, excluding authors with
        # only one sample (they can never contribute a valid positive)
        author_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            author_to_indices[label].append(idx)
 
        self.author_to_indices = {
            author: indices
            for author, indices in author_to_indices.items()
            if len(indices) >= 2
        }
        self.authors = list(self.author_to_indices.keys())
 
        if len(self.authors) < P:
            raise ValueError(
                f"Not enough authors with >=2 samples to fill a batch "
                f"(need P={P}, got {len(self.authors)})"
            )
 
    def __iter__(self):
        # Shuffle authors at the start of each epoch
        authors = self.authors.copy()
        random.shuffle(authors)
 
        batch = []
        for author in authors:
            indices = self.author_to_indices[author]
            # Sample K indices; use replacement if the author has fewer than K samples
            if len(indices) >= self.K:
                chosen = random.sample(indices, self.K)
            else:
                chosen = random.choices(indices, k=self.K)
            batch.extend(chosen)
 
            if len(batch) == self.P * self.K:
                yield batch
                batch = []
        # Drop the last incomplete batch to keep all batches uniform
 
    def __len__(self):
        # Number of complete batches per epoch
        return len(self.authors) // self.P
    
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