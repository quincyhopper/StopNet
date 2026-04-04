import random
import torch
from torch.utils.data import Dataset, Sampler
from datasets import Dataset as HFDataset
from collections import defaultdict

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