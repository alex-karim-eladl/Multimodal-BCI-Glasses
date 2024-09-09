"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/features/wavelets.py

Description: NOT USED
"""

import numpy as np
import pywt

from config import *


def wavelet_packet_decomposition(signal, wavelet, maxlevel):
    wp = pywt.WaveletPacket(signal, wavelet, maxlevel=maxlevel)
    coeffs = []
    # loop over levels 1 through 5 (level 6 is the terminal node)
    for level in range(1, maxlevel - 1):
        coeffs.append([node.data for node in wp.get_level(level, 'natural')])

    return coeffs, wp


# ? stability (means/stds)
def reconstruction_error(X, wavelet, level):
    coeffs = pywt.wavedec(X, wavelet, level=level)
    rec = pywt.waverec(coeffs, wavelet)
    mse = np.mean((X - rec) ** 2)

    """
    X= numpy.genfromtxt('RAW_TIME_DOMAIN_SIGNAL.csv', delimiter=',', skip_header=0)
    d1=[];d2=[];d3=[];d4=[];a4=[]
    main_mse=[]
    for k in range(X.shape[0]):
        a=list(X[k])
        db = pywt.Wavelet('db2')
        coeff = pywt.wavedec(a,db,level=4)
        b=pywt.waverec(coeff, db)
        b=b[0:len(a)]
        mse=0
        for i in range(len(a)):
            mse+=(a[i]-b[i])**2
        main_mse.append((mse**(1/2.00)))
    main_mse=numpy.array(main_mse)
    print(numpy.mean(main_mse))
    """


def wavelets(data, fs, wavelet, maxlevel=8):
    results = []
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel)

    tree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    bandwidth = fs / (2**maxlevel)  # min freq band of maxlevel

    for band, (fmin, fmax) in EEG_BANDS.items():
        bwp = pywt.WaveletPacket(None, wavelet, 'symmetric', maxlevel)
        for i in range(len(tree)):
            # min & max frequencies of the ith band
            bmin = i * bandwidth
            bmax = bmin + bandwidth
            if fmin <= bmin and fmax >= bmax:  # band i within range?
                bwp[tree[i]] = wp[tree[i]].data
        # data corresponding to frequency
        results.append(bwp.reconstruct(update=True))
    return results


# * ============================================================================================== *#
# https://github.com/Coco58323/BCI-IV/blob/601b3e775105d75ad8b50731c3ee20d88be62a32/2_a.py#L109
def feature_bands(x, wavelet='db4', levels=5):
    bands = np.empty((8, x.shape[0], x.shape[1], 225))  # 8 freq band coefficients are chosen from the range 4-32Hz
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            pos = []
            C = pywt.WaveletPacket(x[i, ii, :], wavelet, maxlevel=levels)
            pos = np.append(pos, [node.path for node in C.get_level(5, 'natural')])
            for b in range(1, 9):
                bands[b - 1, i, ii, :] = C[pos[b]].data
    return bands


"""
wpd_data = feature_bands(data)

enc = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1,1)).toarray()

# Cross Validation Split
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score

acc = []
ka = []
prec = []
recall = []

def build_classifier(num_layers = 1):
    classifier = Sequential()
    #First Layer
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu', input_dim = 32,
                        kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu',
                            kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))
    # Last Layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


# # 10-Fold Cross Validation
for train_idx, test_idx in cv.split(labels):

    Csp = [];ss = [];nn = [] # empty lists

    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = X_out[train_idx], X_out[test_idx]

    # CSP filter applied separately for all Frequency band coefficients

    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]
    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x,train_idx,:,:],label_train) for x  in range(8)),axis=-1))

    X_test = ss.transform(np.concatenate(tuple(Csp[x].transform(wpd_data[x,test_idx,:,:]) for x  in range(8)),axis=-1))

    nn = build_classifier()

    nn.fit(X_train, y_train, batch_size = 32, epochs = 300)

    y_pred = nn.predict(X_test)
    pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

    acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))


scores = {'Accuracy':acc,'Kappa':ka,'Precision':prec,'Recall':recall}
Es = pd.DataFrame(scores)
avg = {'Accuracy':[np.mean(acc)],'Kappa':[np.mean(ka)],'Precision':[np.mean(prec)],'Recall':[np.mean(recall)]}
Avg = pd.DataFrame(avg)
T = pd.concat([Es,Avg])
T.index = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','Avg']
T.index.rename('Fold',inplace=True)
print(T)"""


def WPD(signal, wavelet='db4', maxlevel=None):
    wp = pywt.WaveletPacket(signal, wavelet, maxlevel=maxlevel)

    # loop over levels 1 through 5 (level 6 is the terminal node, which we skip)
    coeffs = []
    for level in range(1, 4):
        # print( [node.path for node in wp.get_level(3, 'freq')] ) # sorted by frequency
        level_coeffs = [node.data for node in wp.get_level(level, 'natural')]
        coeffs.append(level_coeffs)

    return coeffs, wp  # return coefficients and wavelet packet object

    #  wp = wt.WaveletPacket(x, wavelet=wv, mode=wpd_mode,maxlevel=wpd_maxlevel)
    # wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
    # WR = np.hstack(wr)
    # nodes = [node for node in wp.get_level(wp.maxlevel, 'natural')]


def wavedec_bands(freq, levels):
    """print frequency bands for wavelet decomposition"""
    if levels == 1:
        return f'0-{freq/2} | {freq/2}-{freq}'
    return f'{wavedec_bands(freq/2, levels-1)} | {freq/2}-{freq}'


def dwt_wavelet(signal, wavelet='db4', level=7, norm=True):
    """
    Compute discrete wavelet transform using pywt.

    Args:
    -----
            signal (dict): raw EEG signal from all channels
            level (int): level of DWT decomposition, default=7 bc of 200Hz sampling frequency
            norm (boolean): normalization by variation in signal, default=True

    Returns:
    --------
            data (array): arrays of coefficients for delta, theta, alpha and beta subbands
    """
    dwt_alpha = []
    dwt_theta = []
    dwt_delta = []
    dwt_beta = []

    for ch in signal:
        dwt = pywt.wavedec(
            ch, wavelet, mode='smooth', level=level
        )  # dwt[0] is approx coeffs dwt[1] means delta subband
        # for band, (low, high) in BANDS.items():
        if norm:
            dwt_beta.append(np.mean(dwt[4] ** 2) / np.var(signal))
            dwt_alpha.append(np.mean(dwt[3] ** 2) / np.var(signal))
            dwt_theta.append(np.mean(dwt[2] ** 2) / np.var(signal))
            dwt_delta.append(np.mean(dwt[1] ** 2) / np.var(signal))
        else:
            dwt_beta.append(np.mean(dwt[4] ** 2))
            dwt_alpha.append(np.mean(dwt[3] ** 2))
            dwt_theta.append(np.mean(dwt[2] ** 2))
            dwt_delta.append(np.mean(dwt[1] ** 2))
    return [dwt_delta, dwt_theta, dwt_alpha, dwt_beta]
