import ot
import os
import torch
import math
import numpy as np

from src.data import dataIterable




def get_pixel_sample(data_iterable, n_samples):
    n_images = len(data_iterable.indicies)
    batch_size = data_iterable.bs
    
    n_batches = math.ceil(n_images/batch_size)
    
    n_samples_per_batch = math.ceil(n_samples/n_batches)
    
    assert n_samples_per_batch > 0
    
    
    samples = torch.tensor([])
    for images, _, _, _ in data_iterable:
        b = torch.transpose(images, 0, 1).flatten(start_dim=1, end_dim=3)
        ids = np.random.randint(low=0, high=b.shape[1], size=n_samples_per_batch)
        samples = torch.cat((samples, b[:, ids]), dim=1)
    
    # remove extra samples
    ids = np.random.randint(low=0, high=samples.shape[1], size=n_samples)
    samples = samples[:, ids]
    return samples


def transport_images(images, transformer):
    shape = images.shape
    for batch in range(shape[0]):
        for row in range(shape[2]):
            images[batch, :, row, :] = transformer.transform(images[batch, :, row, :].T).T


def save_sinkhorn(sinkhorn, path):
    try:
        os.makedirs(path)
    except FileExistsError:
        print(f'Overwriting {path}')
    finally:
        torch.save(sinkhorn.coupling_, os.path.join(path, 'coupling.pt'))
        torch.save(sinkhorn.xt_, os.path.join(path, 'xt.pt'))
        torch.save(sinkhorn.xs_, os.path.join(path, 'xs.pt'))
        
def load_sinkhorn(path, sinkhorn_reg):
    sinkhorn = ot.da.SinkhornTransport(reg_e=sinkhorn_reg)
    sinkhorn.coupling_ = torch.load(os.path.join(path, 'coupling.pt'))
    sinkhorn.xt_ = torch.load(os.path.join(path, 'xt.pt'))
    sinkhorn.xs_ = torch.load(os.path.join(path, 'xs.pt'))
    sinkhorn.nx = sinkhorn._get_backend(sinkhorn.xs_, None, sinkhorn.xt_, None)
    return sinkhorn

def go(dataset, metadata, OT_config):
    
    try:
        load_sinkhorn(
            os.path.join(OT_config['save_path'], f"{OT_config['source_country']}_to_{OT_config['target_country']}"),
            OT_config['reg']
        )
        print("Using Existing Transport.")
    except FileNotFoundError:

        np.random.seed(OT_config['random_seed'])
        
        S = metadata[metadata['country'] == OT_config['source_country']]
        T = metadata[metadata['country'] == OT_config['target_country']]
        
        S_i = S.index.values
        T_i = T.index.values
        
        Xs = get_pixel_sample(dataIterable(dataset, S_i, OT_config['batch_size']), OT_config['n_samples']).T
        Xt = get_pixel_sample(dataIterable(dataset, T_i, OT_config['batch_size']), OT_config['n_samples']).T
        
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=OT_config['reg'])
        ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
        
        save_sinkhorn(
            ot_sinkhorn,
            os.path.join(OT_config['save_path'], f"{OT_config['source_country']}_to_{OT_config['target_country']}")
        )
        