import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA


ANGOLA = 855

fp_ims = 'data/poverty_v1.1/images/'
fp_meta = 'data/poverty_v1.1/dhs_metadata.csv'


ims = np.array([np.load(fp_ims + f'landsat_poverty_img_{x}.npz')['x'] for x in range(ANGOLA)])

with open(fp_meta, encoding="utf-8") as f:
    print(next(f))
    angola_meta = np.array([next(f).split(',') for _ in range(ANGOLA)])

f.close()

angola_wealth = angola_meta[:, 2].astype('float')
    
angola_is_urban = np.array([lst[5]=='True' for lst in angola_meta])

angola_urban_mean = np.mean(np.where(angola_is_urban, angola_meta[:, 2], '0').astype(float))

angola_urban_ims = ims[angola_is_urban].reshape((438, -1))

angola_urban_class = (angola_wealth[angola_is_urban] > angola_urban_mean).astype(int)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(angola_urban_ims)

X = X_pca[:380]
X_test = X_pca[380:]
Y = angola_urban_class[:380]
Y_test = angola_urban_class[380:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
mse = np.mean((clf.predict(X_test) - Y_test)**2)
print(mse)
