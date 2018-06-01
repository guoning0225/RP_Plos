# coding: utf-8

import numpy as np
from utils import *
from config import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import preprocessing, metrics
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import xlrd
import pandas as pd
from pandas import DataFrame
import functools
import h5py
import csv
from collections import Counter


def create_model(cl, args):
    model = Sequential()
    model.add(Dense(32, input_dim=cl, init='normal', activation='relu', kernel_regularizer=regularizers.l2(args.l2)))
    model.add(BatchNormalization())
    model.add(Dropout(args.dropout))
    model.add(Dense(8, input_dim=cl, init='normal', activation='relu', kernel_regularizer=regularizers.l2(args.l2)))
    model.add(BatchNormalization())
    model.add(Dropout(args.dropout))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def plot_model(history, k, j):
    """
        plot the training and testing process of the model
    """
    fig1 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    fig1.savefig(str(k) + '_loss' + str(j) + '.png')
    fig2 = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig2.savefig(str(k) + '_acc' + str(j) + '.png')


def smoteANN_LOO(cl, shuffled_X, shuffled_y, args):
    loo = LeaveOneOut().split(shuffled_X)
    s = []
    test = []
    weights = []
    model = create_model(cl)
    for j, (train_idx, val_idx) in enumerate(loo):
        print('\nFold ', j)
        X_train_cv = shuffled_X[train_idx]
        y_train_cv = shuffled_y[train_idx]
        sm1 = SMOTE(random_state=42)
        X_res, y_res = sm1.fit_sample(X_train_cv, y_train_cv)
        X_valid_cv = shuffled_X[val_idx]
        y_valid_cv = shuffled_y[val_idx]
        history = model.fit(X_res, y_res, validation_data=(X_valid_cv, y_valid_cv), epochs=args.epochs,
                            batch_size=args.batch_size)
        weights.append(get_first_weight(model, cl))
        s.append(model.evaluate(X_valid_cv, y_valid_cv)[1])
        ts = model.predict(X_valid_cv)
        test.append(ts[0][0])
    return s, test, weights


def get_first_weight(model, cl):
    weights = []
    wi = 0
    for i in range(cl):
        wi = np.sum(model.get_weights()[0][i])
        weights.append(wi)
        wi = 0
    return weights


def get_rank(weights, cl):
    ranks = []
    for i in range(len(weights)):
        newweights = weights[i]
        for j in range(len(newweights)):
            newweights[j] = newweights[j]
        b = sorted(enumerate(newweights), reverse=True, key=lambda x: x[1])
        ranks.append(b[:][0:cl // 5])
    with open('test_' + str(cl) + 'fold.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'weight'])
        for i in range(len(ranks)):
            writer.writerows(ranks[i])
            writer.writerow([-1])
    f.close()
    return ranks


def get_summed_weights(weights, cl):
    newweights = [0 for _ in range(cl)]
    for j in range(cl):
        for i in range(len(weights)):
            newweights[j] += weights[i][j]
    with open('weights_' + str(cl) + 'fold.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'weight'])
        for i in range(cl):
            writer.writerow([i + 1, newweights[i]])
    f.close()
    return newweights


def load_dataset(cl, scnt, pcnt, filepath='nadyaFeatureCluster.xlsx', transsheet='filteredName',
                 valuesheet='filteredFeature'):
    data = xlrd.open_workbook(filepath)
    table = data.sheet_by_name(valuesheet)
    cltable = data.sheet_by_name('cluster' + str(cl))
    trans = data.sheet_by_name(transsheet)
    dict = []

    # Get the transformation matrix between clusters and filters
    clusters = [[] for i in range(cl)]
    for i in range(5, 85):
        row = cltable.row_values(i)
        clusters[int(row[4] - 1)].append((int(row[3][2:-1])))
    for i in range(1, trans.nrows):
        row = trans.row_values(i)
        dict[int(row[2])] = row[0][6:]

    # Read in the value
    raw_all = np.zeros([scnt, 80])
    X_all = np.zeros([scnt, cl])
    y_all = np.zeros([scnt])

    # 完成原始特征到聚类特征的映射
    for i in range(pcnt):
        y_all[i] = 1
    for i in range(1, table.nrows):
        raw_all[i - 1] = table.row_values(i, 1)

    for i in range(scnt):
        for j in range(cl):
            for m in clusters[j]:
                X_all[i][j] = X_all[i][j] + raw_all[i][int(dict[m]) - 1]  # -1是为了对齐特征
            X_all[i][j] = X_all[i][j] / len(clusters[j])

    # Shuffle the data
    permutation = np.random.permutation(y_all.shape[0])
    shuffled_X = X_all[permutation, :]
    shuffled_y = y_all[permutation]

    # Normalization
    scaler = preprocessing.StandardScaler().fit(shuffled_X)
    shuffled_X = scaler.transform(shuffled_X)
    return shuffled_X, shuffled_y


if __name__ == "__main__":

    shuffled_X, shuffled_y = load_dataset(cl, scnt, pcnt)
    s, test, weights = smoteANN_LOO(cl, shuffled_X, shuffled_y, args)
    print(weights)
    print(test)
    print(s)
    sum = 0
    for i in s:
        sum = sum + i
    sum = sum / 91
    print(sum)

    get_summed_weights(weights, cl)

    get_rank(weights, cl)
    with open('test_' + str(cl) + 'fold.csv', 'r') as loores:
        reader = csv.reader(loores)
        column = [row[0] for row in reader]
        #     print(column)
    feature = Counter(column).most_common(20)
    feature = feature[1:-1]
    with open('feature_' + str(cl) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'freq'])
        writer.writerows(feature)
