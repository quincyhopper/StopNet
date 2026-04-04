import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections import Counter
import pyarrow.compute as pc
import numpy as np
from datasets import Dataset

def process_batch(batch, stopwords):

    vecs = []
    for text in batch['text']:
        tokens = [tok for tok in word_tokenize(text) if tok in stopwords]
        counter = Counter(tokens)
        vecs.append([counter[w] for w in stopwords])

    return {
        'author': batch['author'],
        'source': batch['source'],
        'embedding': vecs,
        }

def zscore_batch(batch, mean, std):
    normed = (np.array(batch['embedding']) - mean) / std
    return {'embedding': normed.tolist()}

def filter_valid_authors(ds: Dataset, k: int):
    print("Filtering authors")

    author_column = ds.data.column('author')
    value_counts = pc.value_counts(author_column)
    mask = pc.greater_equal(value_counts.field('counts'), k)
    author_counts = value_counts.filter(mask)
    valid_authors = author_counts.field('values')
    full_mask = pc.is_in(ds.data.column('author'), value_set=valid_authors)
    indices = np.where(full_mask.to_numpy())[0]
    
    return ds.select(indices)

if __name__ == "__main__":

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    raw_data = [
        'data/blogtext_raw.parquet'
    ]

    ds = load_dataset(path='parquet', data_files=raw_data, split='train')
    stopwords = sorted(stopwords.words('english'))

    # Authors must have at least 
    filtered_ds = filter_valid_authors(ds, 6)

    # Compute raw frequency vectors
    ds = filtered_ds.map(
        process_batch,
        fn_kwargs={'stopwords': stopwords},
        batched=True,
        batch_size=256,
        num_proc=4,
        remove_columns=['source', 'doc_id']
    )

    print("Computing mean and std for z-score normalisation")
    embeddings = np.array(ds['embedding'])          # (N, D)
    mean = embeddings.mean(axis=0)                  # (D,)
    std  = embeddings.std(axis=0)                   # (D,)
    std  = np.where(std == 0, 1.0, std)             # avoid division by zero for constant features
 
    print("Normalising embeddings")
    ds = ds.map(
        zscore_batch,
        fn_kwargs={'mean': mean, 'std': std},
        batched=True,
        batch_size=256,
        num_proc=4,
    )

    ds.to_parquet('data/blogtext_processed.parquet')

