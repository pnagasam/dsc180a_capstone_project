import os
import numpy as np
import pandas as pd
import torch
from src.data import get_ids, get_clf_cutoffs, dataIterable
from src.OT import load_sinkhorn, transport_images
from src.train import load_best_model



def go(dataset, metadata, eval_config, train_config, OT_config):
    
    np.random.seed(eval_config['random_seed'])
    
    try:
        prev_results = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{eval_config['source_country']}_to_{eval_config['target_country']}.csv"
        ))
    except FileNotFoundError:
        prev_results = None
    
    
    ot_sinkhorn = load_sinkhorn(os.path.join(
        OT_config['save_path'],
        f"{eval_config['source_country']}_to_{eval_config['target_country']}"
    ), OT_config['reg'])
    
    if eval_config['urban']:
        
        S_u_i = get_ids(metadata, eval_config['source_country'], True)
        
        u_model = load_best_model(os.path.join(
            train_config['save_path'],
            eval_config['target_country'],
            'urban'
        ))
        
        u_low_cutoff, u_high_cutoff = get_clf_cutoffs(
            metadata,
            eval_config['target_country'],
            True,
            train_config['low_quantile'],
            train_config['high_quantile']
        )
        
        # eval without transport
        print("Urban Without Transport...")
        actual = np.array([])
        predicted = np.array([])
        ids = np.array([])
        with torch.no_grad():
            for images, labels, meta, idx in dataIterable(dataset, S_u_i, eval_config['batch_size']):

                # classification labels based off on cutoff values
                labels = torch.where(labels < u_low_cutoff, 0, torch.where(labels > u_high_cutoff, 2, 1)).flatten()

                if torch.cuda.is_available():
                        images, labels, meta = images.cuda(), labels.cuda(), meta.cuda()

                outputs = u_model(images, meta)

                actual = np.append(actual, labels.cpu())
                predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
                ids = np.append(ids, idx.cpu())
        res_S_u_wo_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)
        res_S_u_wo_transport = res_S_u_wo_transport.assign(urban=True).assign(transport=False)
        print("Done.")
        
        # eval with transport
        print("Urban With Transport...")
        actual = np.array([])
        predicted = np.array([])
        ids = np.array([])
        with torch.no_grad():
            for images, labels, meta, idx in dataIterable(dataset, S_u_i, eval_config['batch_size']):

                # classification labels based off on cutoff values
                labels = torch.where(labels < u_low_cutoff, 0, torch.where(labels > u_high_cutoff, 2, 1)).flatten()

                transport_images(images, ot_sinkhorn)

                if torch.cuda.is_available():
                        images, labels, meta = images.cuda(), labels.cuda(), meta.cuda()

                outputs = u_model(images, meta)

                actual = np.append(actual, labels.cpu())
                predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
                ids = np.append(ids, idx.cpu())
        res_S_u_wi_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)
        res_S_u_wi_transport = res_S_u_wi_transport.assign(urban=True).assign(transport=True)
        print("Done.")
        
        
    if eval_config['rural']:
        
        S_r_i = get_ids(metadata, eval_config['source_country'], False)
        
        r_model = load_best_model(os.path.join(
            train_config['save_path'],
            eval_config['target_country'],
            'rural'
        ))
        
        r_low_cutoff, r_high_cutoff = get_clf_cutoffs(
            metadata,
            eval_config['target_country'],
            True,
            train_config['low_quantile'],
            train_config['high_quantile']
        )
        
        # eval without transport
        print("Rural Without Transport...")
        actual = np.array([])
        predicted = np.array([])
        ids = np.array([])
        with torch.no_grad():
            for images, labels, meta, idx in dataIterable(dataset, S_r_i, eval_config['batch_size']):

                # classification labels based off on cutoff values
                labels = torch.where(labels < r_low_cutoff, 0, torch.where(labels > r_high_cutoff, 2, 1)).flatten()

                if torch.cuda.is_available():
                        images, labels, meta = images.cuda(), labels.cuda(), meta.cuda()

                outputs = r_model(images, meta)

                actual = np.append(actual, labels.cpu())
                predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
                ids = np.append(ids, idx.cpu())
        res_S_r_wo_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)
        res_S_r_wo_transport = res_S_r_wo_transport.assign(urban=False).assign(transport=False)
        print("Done.")
        
        # eval with transport
        print("Rural With Transport...")
        actual = np.array([])
        predicted = np.array([])
        ids = np.array([])
        with torch.no_grad():
            for images, labels, meta, idx in dataIterable(dataset, S_r_i, eval_config['batch_size']):

                # classification labels based off on cutoff values
                labels = torch.where(labels < r_low_cutoff, 0, torch.where(labels > r_high_cutoff, 2, 1)).flatten()

                transport_images(images, ot_sinkhorn)

                if torch.cuda.is_available():
                        images, labels, meta = images.cuda(), labels.cuda(), meta.cuda()

                outputs = r_model(images, meta)

                actual = np.append(actual, labels.cpu())
                predicted = np.append(predicted, outputs.argmax(dim=1).cpu())
                ids = np.append(ids, idx.cpu())
        res_S_r_wi_transport = pd.DataFrame({'actual': actual, 'predicted': predicted, 'id': ids}).astype(int)
        res_S_r_wi_transport = res_S_r_wi_transport.assign(urban=False).assign(transport=True)
        print("Done.")
        
        
    if eval_config['urban'] and eval_config['rural']:
        transport_results = pd.concat([
            res_S_u_wo_transport,
            res_S_u_wi_transport,
            res_S_r_wo_transport,
            res_S_r_wi_transport,
        ])
    elif prev_results:
        if eval_config['urban']:
            transport_results = pd.concat([
                res_S_u_wo_transport,
                res_S_u_wi_transport,
                prev_results['urban' == False],
            ])
        elif eval_config['rural']:
            transport_results = pd.concat([
                prev_results['urban' == True],
                res_S_r_wo_transport,
                res_S_r_wi_transport,
            ])
        else:
            transport_results = prev_results
    else:
        if eval_config['urban']:
            transport_results = pd.concat([
                res_S_u_wo_transport,
                res_S_u_wi_transport,
            ])
        elif eval_config['rural']:
            transport_results = pd.concat([
                res_S_r_wo_transport,
                res_S_r_wi_transport,
            ])
        else:
            print("Please set either urban or rural to true in config/eval.json")
            
    transport_results.to_csv(
        os.path.join(
            eval_config['save_path'],
            f"{eval_config['source_country']}_to_{eval_config['target_country']}.csv")
    )
    print('Results Saved.')
    