from wilds import get_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import ot
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target-country", default="nigeria")
parser.add_argument("--source-country", default="angola")

args = parser.parse_args()

print(f"Source Country: {args.source_country}")
print(f"Target Country: {args.target_country}")

T_COUNTRY = args.target_country
S_COUNTRY = args.source_country
TRAIN_PROPORTION = .7
VALID_PROPORTION = .2
BATCH_SIZE = 64
N_EPOCHS = 400
LOW_QUANTILE = 1/3
HIGH_QUANTILE = 2/3
RANDOM_SEED = 10
SINKHORN_REG = 1e-1


###    ####   ####   ####   #  #   ####   ####   ####    ##    #  #    ##
#  #   ##     ##      #     ## #    #      #      #     #  #   ## #   #  
#  #   #      #       #     # ##    #      #      #     #  #   # ##      #
###    ####   #      ####   #  #   ####    #     ####    ##    #  #    ##

def tvt_split_index(ids, train_p, valid_p):
    n = len(ids)
    np.random.shuffle(ids)
    train = ids[:int(n*train_p)]
    valid = ids[int(n*train_p):int(n*(train_p+valid_p))]
    test = ids[int(n*(train_p+valid_p)):]
    return train, valid, test


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
            return tuple([torch.stack([dataset[d][q] for d in range(i, min(i+self.bs, len(self.indicies)))]) for q in range(3)]+[torch.tensor(self.indicies[i:min(i+self.bs, len(self.indicies))])])
            
        else:
            raise StopIteration
        



# Neural Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(8, 48, 3)
        self.conv2 = nn.Conv2d(48, 96, 3)
        self.conv3 = nn.Conv2d(96, 1, 1)
        
        self.fc1 = nn.Linear(2916, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x, meta):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# function to log training info
def log(train_loss, valid_loss, train_acc, valid_acc, to):
    new_row = pd.DataFrame({
        'train_loss': [train_loss],
         'valid_loss': [valid_loss],
         'train_acc': [train_acc],
         'valid_acc': [valid_acc]
    })
    
    return pd.concat([to, new_row], ignore_index=True)



def train_model(train_ids, valid_ids, n_epochs, low_cutoff, high_cutoff, save_path):
    
    try: # read in previous best model

        training_info = pd.read_csv(os.path.join(save_path, 'training_info.csv'))

        print("Previous Training Data Found.")
        print("Loading Best Model...")

        best_model_id = training_info['valid_loss'].idxmin()
        best_model = torch.load(os.path.join(save_path, f'CNN_epoch_{best_model_id:03d}'))

    except FileNotFoundError:

        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass
        
        print("No Previous Data Found.")
        print("Beginning Training...")

        net = Net()

        # put NN on GPU
        if torch.cuda.is_available():
            net = net.cuda()


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters())

        # data structure for logging training info
        training_info = pd.DataFrame(
            {'train_loss': [],
            'valid_loss': [],
            'train_acc': [],
            'valid_acc': []
            }
        )

        for epoch in range(n_epochs):

            running_loss = 0.0
            batches = 0
            correct = 0
            count = 0
            for images, labels, metadata, _ in dataIterable(dataset, train_ids, BATCH_SIZE):
          
                # classification labels based off on borders
                labels = torch.where(labels < low_cutoff, 0, torch.where(labels > high_cutoff, 2, 1)).flatten()

                # put data onto gpu if available
                if torch.cuda.is_available():
                    images, labels, metadata = images.cuda(), labels.cuda(), metadata.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images, metadata)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                batches += 1
                correct += (labels == outputs.argmax(dim=1)).sum().item()
                count += outputs.shape[0]

            train_loss = running_loss/batches
            train_acc = correct/count
            
            running_loss = 0.0
            batches = 0
            correct = 0
            count = 0
            with torch.no_grad():
                for images, labels, metadata, _ in dataIterable(dataset, valid_ids, BATCH_SIZE):
                    
                    # classification labels based off on borders
                    labels = torch.where(labels < low_cutoff, 0, torch.where(labels > high_cutoff, 2, 1)).flatten()
                    
                    # put data onto gpu if available
                    if torch.cuda.is_available():
                        images, labels, metadata = images.cuda(), labels.cuda(), metadata.cuda()
                        
                    outputs = net(images, metadata)
                    
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    batches += 1
                    correct += (labels == outputs.argmax(dim=1)).sum().item()
                    count += outputs.shape[0]
            valid_loss = running_loss/batches
            valid_acc = correct/count

            training_info = log(train_loss, valid_loss, train_acc, valid_acc, training_info)

            torch.save(net, os.path.join(save_path, f'CNN_epoch_{epoch:03d}'))

            print(training_info.iloc[[-1]])
            
            # if stuck
            if epoch > 10:
                if training_info['train_acc'].iloc[-1] < .9:
                    if training_info['train_loss'].iloc[-10:].round(4).nunique() == 1:
                        net = Net()                                                    # re init weights
                        optimizer = optim.AdamW(net.parameters())
                        
                        

        print("Training Done.")

        print("Saving Training Data...")
        training_info.to_csv(os.path.join(save_path, 'training_info.csv'))

        print("Loading Best Model...")
        best_model_id = training_info['valid_loss'].idxmin()
        best_model = torch.load(os.path.join(save_path, f'CNN_epoch_{best_model_id:03d}'))

    finally:
        return best_model
    

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
        os.mkdir(path)
    except FileExistsError:
        print(f'Overwriting {path}')
    finally:
        torch.save(sinkhorn.coupling_, os.path.join(path, 'coupling.pt'))
        torch.save(sinkhorn.xt_, os.path.join(path, 'xt.pt'))
        torch.save(sinkhorn.xs_, os.path.join(path, 'xs.pt'))
        
def load_sinkhorn(path):
    sinkhorn = ot.da.SinkhornTransport(reg_e=SINKHORN_REG)
    sinkhorn.coupling_ = torch.load(os.path.join(path, 'coupling.pt'))
    sinkhorn.xt_ = torch.load(os.path.join(path, 'xt.pt'))
    sinkhorn.xs_ = torch.load(os.path.join(path, 'xs.pt'))
    sinkhorn.nx = sinkhorn._get_backend(sinkhorn.xs_, None, sinkhorn.xt_, None)
    return sinkhorn
            
#                                                   #
# - - - Data Preprocessing and Model Training - - - #
#                                                   #



np.random.seed(RANDOM_SEED)


dataset = get_dataset(dataset="poverty", download=False)


# target country
T_country = T_COUNTRY

# set train, valid proportions
# test is the rest
train_p, valid_p = TRAIN_PROPORTION, VALID_PROPORTION

meta = pd.read_csv('../../notebooks/data/poverty_v1.1/dhs_metadata.csv')

T = meta[meta['country'] == T_country]

# Urban and rural split
T_u = T[T['urban'] == True]
T_r = T[T['urban'] == False]

T_u_i = T_u.index.values
T_r_i = T_r.index.values

T_u_i_train, T_u_i_valid, T_u_i_test = tvt_split_index(T_u_i, train_p, valid_p)
T_r_i_train, T_r_i_valid, T_r_i_test = tvt_split_index(T_r_i, train_p, valid_p)




# classification quantiles
low_quantile = LOW_QUANTILE
high_quantile = HIGH_QUANTILE

# calculate borders based off quantile
u_low_cutoff, u_high_cutoff = T_u['wealthpooled'].quantile(low_quantile), T_u['wealthpooled'].quantile(high_quantile)
r_low_cutoff, r_high_cutoff = T_r['wealthpooled'].quantile(low_quantile), T_r['wealthpooled'].quantile(high_quantile)

    
batch_size = BATCH_SIZE

### ../models/rural

# Train on Target Domain

print("Urban:")
T_u_model = train_model(
    T_u_i_train,
    T_u_i_valid,
    N_EPOCHS,
    u_low_cutoff,
    u_high_cutoff,
    f'../../models/{T_COUNTRY}/urban'
)

print("Rural:")
T_r_model = train_model(
    T_r_i_train,
    T_r_i_valid,
    N_EPOCHS,
    r_low_cutoff,
    r_high_cutoff,
    f'../../models/{T_COUNTRY}/rural'
)



#                               #
# - - - Optimal Transport - - - #
#                               #

# source country
S_country = S_COUNTRY

S = meta[meta['country'] == S_country]

# Urban and rural split
S_u = S[S['urban'] == True]
S_r = S[S['urban'] == False]

# indicies
S_u_i = S_u.index.values
S_r_i = S_r.index.values


# evaluate without transport
print('Evaulating Without Transport...')
print('Urban')
# urban
actual = np.array([])
predicted = np.array([])
ids = np.array([])
with torch.no_grad():
    for images, labels, metadata, idx in dataIterable(dataset, S_u_i, BATCH_SIZE):
        
        # classification labels based off on cutoff values
        labels = torch.where(labels < u_low_cutoff, 0, torch.where(labels > u_high_cutoff, 2, 1)).flatten()
        
        if torch.cuda.is_available():
                images, labels, meta = images.cuda(), labels.cuda(), metadata.cuda()
        
        outputs = T_u_model(images, meta)

        actual = np.append(actual, labels.cpu())
        predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
        ids = np.append(ids, idx.cpu())
res_S_u_wo_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)

print('Rural')
# rural
actual = np.array([])
predicted = np.array([])
ids = np.array([])
with torch.no_grad():
    for images, labels, metadata, idx in dataIterable(dataset, S_r_i, BATCH_SIZE):
        
        # classification labels based off on cutoff values
        labels = torch.where(labels < r_low_cutoff, 0, torch.where(labels > r_high_cutoff, 2, 1)).flatten()
        
        if torch.cuda.is_available():
                images, labels, meta = images.cuda(), labels.cuda(), metadata.cuda()
        
        outputs = T_r_model(images, meta)

        actual = np.append(actual, labels.cpu())
        predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
        ids = np.append(ids, idx.cpu())
res_S_r_wo_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)






# transport
print('Computing Transport...')
# urban
Xs = get_pixel_sample(dataIterable(dataset, S_u_i, 10), 500).T
Xt = get_pixel_sample(dataIterable(dataset, T_u_i, 10), 500).T

ot_sinkhorn = ot.da.SinkhornTransport(reg_e=SINKHORN_REG)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

print('Saving Sinkhorn Transport...')
save_sinkhorn(ot_sinkhorn, f'../../results/{S_COUNTRY}_to_{T_COUNTRY}')

print('Evaluating With Optimal Transport...')
# evaluate with transport

print('Urban')
# urban
actual = np.array([])
predicted = np.array([])
ids = np.array([])
with torch.no_grad():
    for images, labels, metadata, idx in dataIterable(dataset, S_u_i, BATCH_SIZE):
        
        # classification labels based off on cutoff values
        labels = torch.where(labels < u_low_cutoff, 0, torch.where(labels > u_high_cutoff, 2, 1)).flatten()
        
        transport_images(images, ot_sinkhorn)
        
        if torch.cuda.is_available():
                images, labels, meta = images.cuda(), labels.cuda(), metadata.cuda()
        
        outputs = T_u_model(images, meta)
        
        actual = np.append(actual, labels.cpu())
        predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
        ids = np.append(ids, idx.cpu())
res_S_u_wi_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)


print('Rural')
# urban
actual = np.array([])
predicted = np.array([])
ids = np.array([])
with torch.no_grad():
    for images, labels, metadata, idx in dataIterable(dataset, S_r_i, BATCH_SIZE):
        
        # classification labels based off on cutoff values
        labels = torch.where(labels < r_low_cutoff, 0, torch.where(labels > r_high_cutoff, 2, 1)).flatten()
        
        transport_images(images, ot_sinkhorn)
        
        if torch.cuda.is_available():
                images, labels, meta = images.cuda(), labels.cuda(), metadata.cuda()
        
        outputs = T_r_model(images, meta)
        
        actual = np.append(actual, labels.cpu())
        predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
        ids = np.append(ids, idx.cpu())
res_S_r_wi_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)

# save results

res_S_u_wo_transport = res_S_u_wo_transport.assign(urban=True).assign(transport=False)

res_S_r_wo_transport = res_S_r_wo_transport.assign(urban=False).assign(transport=False)

res_S_u_wi_transport = res_S_u_wi_transport.assign(urban=True).assign(transport=True)

res_S_r_wi_transport = res_S_r_wi_transport.assign(urban=False).assign(transport=True)

transport_results = pd.concat([
    res_S_u_wo_transport,
    res_S_r_wo_transport,
    res_S_u_wi_transport,
    res_S_r_wi_transport,
])

print('Saving Results...')
transport_results.to_csv(f'../../results/{S_COUNTRY}_to_{T_COUNTRY}.csv')