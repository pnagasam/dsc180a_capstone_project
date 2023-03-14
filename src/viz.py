import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.OT import load_sinkhorn, transport_images
import os
    


def go(dataset, metadata, viz_config, train_config, eval_config, OT_config):

    np.random.seed(viz_config['random_seed'])

    def show_channels_id(id, figsize=(30, 15)):
        fig, axs = plt.subplots(2, 4, figsize=figsize)

        channels = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

        for i in range(8):
            axs[i//4][i%4].imshow(dataset[id][0][i])
            axs[i//4][i%4].set_title(channels[i])

        print(f'wealth index: {dataset[id][1][0]}')

    def show_channels_im(im, figsize=(30, 15)):
        fig, axs = plt.subplots(2, 4, figsize=figsize)

        channels = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

        for i in range(8):
            axs[i//4][i%4].imshow(im[0][i])
            axs[i//4][i%4].set_title(channels[i])
            
    def show_image(id, title=None):
        # definitely not accurate...
        plt.imshow((torch.stack([dataset[id][0][2], dataset[id][0][1], dataset[id][0][0]], dim=2)+1.2290)/(2.6758+1.2290))
        plt.title(title)

        print(f'wealth index: {dataset[id][1][0]}')
        
    def compare_ot_and_not(imid, sinkhorn):

        #print("Transporting image...")
        im = dataset[imid][0][None, :, :, :].clone()
        transport_images(im, sinkhorn)

        print("OT")
        plt.imshow((torch.stack([im[0][2], im[0][1], im[0][0]], dim=2)+1.2290)/(2.6758+1.2290))
        plt.title("RGB")
        plt.show()
        print("no OT")
        show_image(imid, title="RGB")
        plt.show()
        print("OT")
        show_channels_im(im)
        plt.show()
        print("no OT")
        show_channels_id(imid)
        plt.show()
    
    plt.rcParams.update({'font.size': 15})
    
    
    if viz_config['asset_index_dist']:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        u = metadata[metadata['urban'] == True]
        r = metadata[metadata['urban'] == False]

        ax.hist(u['wealthpooled'], bins=100, alpha=0.6, color = 'tab:blue', label='urban')
        ax.hist(r['wealthpooled'], bins=100, alpha=0.6, color = 'tab:orange', label='rural')
        ax.legend()
        ax.set_xlabel('Asset index')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Asset Index')
        fig.show()
    
    
    if viz_config['clf_cutoffs']:
        # classification quantiles
        low_quantile = viz_config['low_quantile']
        high_quantile = viz_config['high_quantile']

        u = metadata[metadata['urban'] == True]
        r = metadata[metadata['urban'] == False]
        
        T_u = u[u['country'] == viz_config['target_country']]
        T_r = r[r['country'] == viz_config['target_country']]
        
        # calculate borders based off quantile
        u_low_cutoff, u_high_cutoff = T_u['wealthpooled'].quantile(low_quantile), T_u['wealthpooled'].quantile(high_quantile)
        r_low_cutoff, r_high_cutoff = T_r['wealthpooled'].quantile(low_quantile), T_r['wealthpooled'].quantile(high_quantile)



        #                     #
        # ---- make plot ---- #
        #                     #

        # subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # histogram
        axs[1].hist(T_u['wealthpooled'], bins=30, alpha=0.5, label='urban')
        axs[0].hist(T_r['wealthpooled'], bins=30, alpha=0.5, color='tab:orange', label='rural')

        # low cutoff
        axs[1].axvline(x=u_low_cutoff, color='blue', label=f'{round(low_quantile*100, 1)} Percentile\n{round(u_low_cutoff, 2)} Asset index')
        axs[0].axvline(x=r_low_cutoff, color='blue', label=f'{round(low_quantile*100, 1)} Percentile\n{round(r_low_cutoff, 2)} Asset index')

        # high cutoff
        axs[1].axvline(x=u_high_cutoff, color='red', label=f'{round(high_quantile*100, 1)} Percentile\n{round(u_high_cutoff, 2)} Asset index')
        axs[0].axvline(x=r_high_cutoff, color='red', label=f'{round(high_quantile*100, 1)} Percentile\n{round(r_high_cutoff, 2)} Asset index')

        # legend
        axs[1].legend()
        axs[0].legend()

        # subplot titles
        axs[1].set_title("Urban")
        axs[0].set_title("Rural")

        # subplot x labels
        axs[1].set_xlabel("Asset index")
        axs[0].set_xlabel("Asset index")

        fig.suptitle("Classification Cutoffs", fontsize='25')
        fig.show()
        
    if viz_config['training_info']['urban']:
        t = pd.read_csv(os.path.join(
            train_config['save_path'],
            viz_config['target_country'],
            'urban',
            'training_info.csv'
        ))
        t.iloc[:t['valid_loss'].idxmin()].plot(figsize=(6.5, 4.5), xlabel='epoch', title=f"{viz_config['target_country']} urban training info")
        
    if viz_config['training_info']['rural']:
        t = pd.read_csv(os.path.join(
            train_config['save_path'],
            viz_config['target_country'],
            'rural',
            'training_info.csv'
        ))
        t.iloc[:t['valid_loss'].idxmin()].plot(figsize=(6.5, 4.5), xlabel='epoch', title=f"{viz_config['target_country']} rural training info")
        
        
    if viz_config['source_confusion_matrix']['urban']['without_OT']:
        df = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}.csv"
        ))
        r = df[df['urban'] == True]
        t = r[r['transport'] == False]
        t = t.replace([2, 1, 0], ["poor", 'moderate', 'wealthy'])
        p = t.pivot_table(index='predicted', columns='actual', values='urban', aggfunc='count')
        p = p.fillna(0).astype(int).loc[['wealthy', 'moderate', 'poor']][['wealthy', 'moderate', 'poor']]
        print("Confusion matrix of source domain (urban, without OT):")
        print(p.head())
        print()
        
    if viz_config['source_confusion_matrix']['urban']['with_OT']:
        df = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}.csv"
        ))
        r = df[df['urban'] == True]
        t = r[r['transport'] == True]
        t = t.replace([2, 1, 0], ["poor", 'moderate', 'wealthy'])
        p = t.pivot_table(index='predicted', columns='actual', values='urban', aggfunc='count')
        p = p.fillna(0).astype(int).loc[['wealthy', 'moderate', 'poor']][['wealthy', 'moderate', 'poor']]
        print("Confusion matrix of source domain (urban, with OT):")
        print(p.head())
        print()
        
    if viz_config['source_confusion_matrix']['rural']['without_OT']:
        df = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}.csv"
        ))
        r = df[df['urban'] == False]
        t = r[r['transport'] == False]
        t = t.replace([2, 1, 0], ["poor", 'moderate', 'wealthy'])
        p = t.pivot_table(index='predicted', columns='actual', values='urban', aggfunc='count')
        p = p.fillna(0).astype(int).loc[['wealthy', 'moderate', 'poor']][['wealthy', 'moderate', 'poor']]
        print("Confusion matrix of source domain (rural, without OT):")
        print(p.head())
        print()
        
    if viz_config['source_confusion_matrix']['rural']['with_OT']:
        df = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}.csv"
        ))
        r = df[df['urban'] == False]
        t = r[r['transport'] == True]
        t = t.replace([2, 1, 0], ["poor", 'moderate', 'wealthy'])
        p = t.pivot_table(index='predicted', columns='actual', values='urban', aggfunc='count')
        p = p.fillna(0).astype(int).loc[['wealthy', 'moderate', 'poor']][['wealthy', 'moderate', 'poor']]
        print("Confusion matrix of source domain (rural, with OT):")
        print(p.head())
        print()
        
        
    if viz_config['show_changed']:
        
        ot_sinkhorn = load_sinkhorn(os.path.join(
            OT_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}"
        ), OT_config['reg'])
        
        df = pd.read_csv(os.path.join(
            eval_config['save_path'],
            f"{viz_config['source_country']}_to_{viz_config['target_country']}.csv"
        ))
        
        df = df.replace([2, 1, 0], ["poor", 'moderate', 'wealthy'])
        df = df.pivot_table(index='id', columns='transport', values=['predicted', 'actual'], aggfunc=sum)
        df = df[df['predicted'][False] != df['predicted'][True]]
        a = df['actual'][False]
        p = pd.DataFrame({
            'actual': df['actual'][False],
            'without_OT': df['predicted'][False],
            'with_OT': df['predicted'][True]
        })

        s = p.sample()
        print(s.head())
        imid = s.index.values[0]
        
        compare_ot_and_not(imid, ot_sinkhorn)
        