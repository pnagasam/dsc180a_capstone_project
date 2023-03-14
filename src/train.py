import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from src.data import *


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


def load_best_model(save_path):
    
    training_info = pd.read_csv(os.path.join(save_path, 'training_info.csv'))
    
    best_model_id = training_info['valid_loss'].idxmin()
    best_model = torch.load(os.path.join(save_path, f'CNN_epoch_{best_model_id:03d}'))
    
    return best_model
    

def log(train_loss, valid_loss, train_acc, valid_acc, to):
    new_row = pd.DataFrame({
        'train_loss': [train_loss],
         'valid_loss': [valid_loss],
         'train_acc': [train_acc],
         'valid_acc': [valid_acc]
    })
    
    return pd.concat([to, new_row], ignore_index=True)



def train_model(dataset, train_ids, valid_ids, n_epochs, batch_size, low_cutoff, high_cutoff, save_path):
    
    try: # read in previous best model
        
        best_model = load_best_model(save_path)

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
            print("CUDA Available!")
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
            for images, labels, metadata, _ in dataIterable(dataset, train_ids, batch_size):
          
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
                for images, labels, metadata, _ in dataIterable(dataset, valid_ids, batch_size):
                    
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
                        net = Net()  # re init weights
                        if torch.cuda.is_available():
                            net = net.cuda()
                        optimizer = optim.AdamW(net.parameters())
                        
                        

        print("Training Done.")

        print("Saving Training Data...")
        training_info.to_csv(os.path.join(save_path, 'training_info.csv'))

        print("Loading Best Model...")
        best_model_id = training_info['valid_loss'].idxmin()
        best_model = torch.load(os.path.join(save_path, f'CNN_epoch_{best_model_id:03d}'))

    finally:
        return best_model

    
    
    
    
    
def go(dataset, metadata, train_config):
    
    np.random.seed(train_config['random_seed'])

    if train_config['urban']:


        T_u_i_train, T_u_i_valid, T_u_i_test = tvt_split_index(
            get_ids(metadata, train_config['country'], urban=True),
            train_config['train_proportion'],
            train_config['valid_proportion']
        )


        u_low_cutoff, u_high_cutoff = get_clf_cutoffs(
            metadata,
            train_config['country'],
            True,
            train_config['low_quantile'],
            train_config['high_quantile']
        )


        u_model = train_model(
            dataset,
            T_u_i_train, T_u_i_valid,
            train_config['n_epochs'],
            train_config['batch_size'],
            u_low_cutoff, u_high_cutoff,
            os.path.join(train_config['save_path'], train_config['country'], 'urban')
        )

    if train_config['rural']:

        T_r_i_train, T_r_i_valid, T_r_i_test = tvt_split_index(
            get_ids(metadata, train_config['country'], urban=False),
            train_config['train_proportion'],
            train_config['valid_proportion']
        )


        r_low_cutoff, r_high_cutoff = get_clf_cutoffs(
            metadata,
            train_config['country'],
            True,
            train_config['low_quantile'],
            train_config['high_quantile']
        )

        r_model = train_model(
            dataset,
            T_u_i_train, T_u_i_valid,
            train_config['n_epochs'],
            train_config['batch_size'],
            r_low_cutoff, r_high_cutoff,
            os.path.join(train_config['save_path'], train_config['country'], 'rural')
        )
        
        
        