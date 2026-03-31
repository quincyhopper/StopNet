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

    raw_data = [
        'data/blogtext_raw.parquet'
    ]

    ds = load_dataset(path='parquet', data_files=raw_data, split='train')
    stopwords = set(stopwords.words('english'))

    # Authors must have at least 
    filtered_ds = filter_valid_authors(ds, 6)

    ds = filtered_ds.map(
        process_batch,
        fn_kwargs={'stopwords': stopwords},
        batched=True,
        batch_size=256,
        num_proc=4,
        remove_columns=['author', 'text', 'source', 'doc_id']
    )

    ds.to_parquet('data/blogtext_processed.parquet')

