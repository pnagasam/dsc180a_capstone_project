import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
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

def train_model(train_country):

    assert train_country in COUNTRIES

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

    print('creating DataFrame')

    df_data = []
    for label, loc, nl_mean, nl_center in zip(labels, locs, nls_mean, nls_center):
        lat, lon = loc
        loc_info = loc_dict[(lat, lon)]
        country = loc_info['country']
        year = int(loc_info['country_year'][-4:])  # use the year matching the surveyID
        urban = loc_info['urban']
        household = loc_info['households']
        row = [lat, lon, label, country, year, urban, nl_mean, nl_center, household]
        df_data.append(row)
    df = pd.DataFrame.from_records(
        df_data,
        columns=['lat', 'lon', 'wealthpooled', 'country', 'year', 'urban', 'nl_mean', 'nl_center', 'households'])

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

    image_hists = image_hists/(np.max(image_hists)*1.0)

    print('reshaping train')

    idxs = country_labels==COUNTRIES.index(train_country)
    angola_hists = image_hists[idxs]
    angola_hists_flat = angola_hists.reshape((angola_hists.shape[0], angola_hists.shape[1]*angola_hists.shape[2]))
    angola_wealth = wealth_labels[idxs]


    ntrain = 9*idxs.sum()//10
    X_train = angola_hists_flat[:ntrain,:]
    X_test = angola_hists_flat[ntrain:,:]
    y_train = angola_wealth[:ntrain]
    y_test = angola_wealth[ntrain:]


    print('building classifier')

    clf = RandomForestClassifier(max_depth=3).fit(X_train, y_train)

    print('train model')

    mse_train = np.mean((y_train - clf.predict(X_train))**2)
    mse_test = np.mean((y_test - clf.predict(X_test))**2)

    print(f'{train_country} mse train: {mse_train}\n{train_country} mse test:  {mse_test}')

    return clf, angola_hists