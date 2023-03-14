from wilds import get_dataset
import pandas as pd
import os
import numpy as np
import torch

def load_dataset():
    dataset = get_dataset(dataset="poverty", download=True)
    metadata = pd.read_csv('data/poverty_v1.1/dhs_metadata.csv')
    return dataset, metadata

def tvt_split_index(ids, train_p, valid_p):
    n = len(ids)
    np.random.shuffle(ids)
    train = ids[:int(n*train_p)]
    valid = ids[int(n*train_p):int(n*(train_p+valid_p))]
    test = ids[int(n*(train_p+valid_p)):]
    return train, valid, test

def get_ids(metadata, country, urban):
    ids = metadata[metadata['country'] == country]
    ids = ids[ids['urban'] == urban]
    ids = ids.index.values
    return ids

def get_clf_cutoffs(metadata, country, urban, low_quantile, high_quantile):
    c = metadata[metadata['country'] == country]
    u = c[c['urban'] == urban]
    low_cutoff = u['wealthpooled'].quantile(low_quantile)
    high_cutoff = u['wealthpooled'].quantile(high_quantile)
    return low_cutoff, high_cutoff

class dataIterable:
    def __init__(self, dataset, indicies, batch_size=1):
        
        self.dataset = dataset
        self.indicies = indicies
        self.i = 0
        self.bs = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i < len(self.indicies):
            
            i = self.i
            self.i += self.bs
            return tuple([torch.stack([self.dataset[d][q] for d in range(i, min(i+self.bs, len(self.indicies)))]) for q in range(3)]+[torch.tensor(self.indicies[i:min(i+self.bs, len(self.indicies))])])
            
        else:
            raise StopIteration