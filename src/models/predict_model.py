import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import ot
import glob
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests

hist_data_path = './data/dhs_image_hists.npz'
hist_data_url = 'https://github.com/sustainlab-group/africa_poverty/blob/3d820ffc399f8c994b4b8da563bf09a7fc37defa/data/dhs_image_hists.npz?raw=true'

loc_data_path = './data/dhs_loc_dict.pkl'
loc_data_url = 'https://github.com/sustainlab-group/africa_poverty/blob/3d820ffc399f8c994b4b8da563bf09a7fc37defa/data/dhs_loc_dict.pkl?raw=true'

COUNTRIES = ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
       'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea',
       'kenya', 'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria',
       'rwanda', 'senegal', 'sierra_leone', 'tanzania', 'togo', 'uganda',
       'zambia', 'zimbabwe']

def predict_model(clf, predict_country, train_hists):

    assert predict_country in COUNTRIES

    if os.path.exists(hist_data_path) and os.path.exists(loc_data_path):
        print('data found')

    else:

        if not os.path.exists('./data'):
            os.mkdir('./data')

        print('downloading hist data')

        r = requests.get(hist_data_url)

        print('writing hist data')

        f = open(hist_data_path,'wb')

        f.write(r.content)

        print('downloading loc data')

        r = requests.get(loc_data_url)

        print('writing loc data')

        f = open(loc_data_path, 'wb')

        f.write(r.content)


    print('importing loc data')

    with open(loc_data_path, 'rb') as f:
        loc_dict = pickle.load(f)

    print('importing hist data')

    npz = np.load(hist_data_path)

    image_hists = npz['image_hists'][:, :-1, :]
    labels = npz['labels']
    locs = npz['locs']
    years = npz['years']
    nls_center = npz['nls_center']
    nls_mean = npz['nls_mean']

    print('extracting labels')

    country_indices = defaultdict(list)  # country => np.array of indices
    country_labels = np.zeros(len(locs), dtype=np.int32)  # np.array of country labels
    wealth_labels = np.zeros(len(locs), dtype=np.float32)

    for i, loc in enumerate(locs):
        country = loc_dict[tuple(loc)]['country']
        wealth_labels[i] = loc_dict[tuple(loc)]['wealthpooled']>0
        country_indices[country].append(i)

    for i, country in enumerate(COUNTRIES):
        country_indices[country] = np.asarray(country_indices[country])
        indices = country_indices[country]
        country_labels[indices] = i

    print('reshapeing test')

    idxs = country_labels==COUNTRIES.index(predict_country)
    benin_hists = image_hists[idxs]
    benin_hists_flat = benin_hists.reshape((benin_hists.shape[0], benin_hists.shape[1]*benin_hists.shape[2]))
    benin_wealth = wealth_labels[idxs]
    benin_hists_flat.shape, benin_wealth.shape

    image_hists = image_hists/(np.max(image_hists)*1.0)

    print('')

    Xt = train_hists[0].T
    Xs = benin_hists[0].T
    N = 10
    for i in range(1, N):
        Xt = np.where(train_hists[i].T > Xt, train_hists[i].T, Xt)
        Xs = np.where(benin_hists[i].T > Xs, benin_hists[i].T, Xs)

    print('computing sinkhorn OT')

    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=benin_hists[0].T)
    transp_Xs_sinkhorn[transp_Xs_sinkhorn<0] = 0

    print('building test')

    X = np.full_like(benin_hists_flat, 0)
    y = benin_wealth

    for i in range(X.shape[0]):
        trans = ot_sinkhorn.transform(benin_hists[i].T)
        trans[trans<0]=0
        X[i] = trans.reshape(trans.shape[0]*trans.shape[1])

    print(f'{predict_country} mse with OT:    {np.mean((y - clf.predict(X))**2)}\n{predict_country} mse without OT: {np.mean((y - clf.predict(benin_hists_flat))**2)}')