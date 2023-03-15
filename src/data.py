from wilds import get_dataset
import pandas as pd
import os
import numpy as np
import torch

COUNTRIES = ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire', 'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya', 'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal', 'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']


def load_dataset():
    dataset = get_dataset(dataset="poverty", download=True)
    metadata = pd.read_csv('data/poverty_v1.1/dhs_metadata.csv')
    return dataset, metadata

def generate_dummmy_data(size):
    def d_0():
            return torch.rand(8, 224, 224)
    def d_1():
        return torch.randint(3, (1, 1))[0]
    def d_2(): 
        return torch.cat([torch.randint(2, (1, 1)), torch.rand(1, 1), torch.randint(3, (1, 1)), torch.tensor([[0]])]).T[0]

    dataset = [(d_0(), d_1(), d_2()) for _ in range(size)]

    metadata = pd.DataFrame()
    metadata['lat'] = (np.random.rand(size)*60)-30
    metadata['lon'] = (np.random.rand(size)*60)-30
    metadata['wealthpooled'] = (np.random.rand(size)*3)-1.5
    metadata['country'] = np.random.choice(COUNTRIES, size)
    metadata['year'] = np.random.choice(list(range(2009, 2017)), size)
    metadata['urban'] = np.random.choice([True, False], size)
    metadata['nl_mean'] = (np.random.rand(size)*20)-2
    metadata['nl_center'] = (np.random.rand(size)*20)-2
    metadata['households'] = np.random.randint(0, 50, size)

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