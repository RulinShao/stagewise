import numpy as np
import pandas as pd
import os
import time 
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import FastICA
# from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random

from sythetic import MultimodalDataset, get_intersections
from independent_stagewise import (
    get_res_y, 
    get_res_y_2, 
    add_res_y, 
    add_res_y_2,
)


seed = 10
random.seed(seed)
np.random.seed(seed)


def calculate_accuracy(y_pred, y):
    y_pred, y = torch.tensor(y_pred), torch.tensor(y)
    pred = (y_pred >= 0.5).float()
    acc = (pred == y).sum() / y.shape[0]
    return acc.item()

def mean_squared_error(pred, act):
    pred, act = torch.tensor(pred), torch.tensor(act)
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff.item()

def get_corr(X, y):
    corr, _ = pearsonr(X, y)
    return corr

def normalization(data, normalization_method='uniform', data_type='df'):
    if data_type == 'df':
        if normalization_method == 'uniform':  # into range [0, 1]
            for column in data.columns:
                data[column] = (data[column] -  data[column].min()) / (data[column].max() - data[column].min())
        else:
            for column in data.columns:  # into N(0, 1)
                data[column] = (data[column] -
                                    data[column].mean()) / data[column].std()  
    else:
        if normalization_method == 'uniform':  # into range [0, 1]
            data = (data - np.amin(data, axis=0)) / (np.amax(data, axis=0) - np.amin(data, axis=0))
        else:
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  
    return data

########################## Load Raw Data ##########################
df = pd.read_csv(os.path.join('~/data/', 'HomeC.csv'), low_memory=False)
print(df.head(), '\n', df.columns)

########################## Data Preprocessing ##########################
# Rename columns to remove spaces and the kW unit 
df.columns = [col[:-5].replace(' ','_') if 'kW' in col else col for col in df.columns]

# Drop rows with nan values 
df = df.dropna()

# The columns "use" and "house_overall" are the same, so let's remove the 'house_overall' column
df.drop(['House_overall'], axis=1, inplace=True)

# The columns "gen" and "solar" are the same, so let's remove the 'solar' column
df.drop(['Solar'], axis=1, inplace=True)

# drop rows with cloudCover column values that are not numeric (bug in sensors) and convert column to numeric
df = df[df['cloudCover']!='cloudCover']
df["cloudCover"] = pd.to_numeric(df["cloudCover"])

# Create columns that regroup kitchens and furnaces 
df['kitchen'] = df['Kitchen_12'] + df['Kitchen_14'] + df['Kitchen_38']
df['Furnace'] = df['Furnace_1'] + df['Furnace_2']

# Convert "time" column (which is a unix timestamp) to a Y-m-d H-M-S 
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(df['time'].iloc[0])))
time_index = pd.date_range(start_time, periods=len(df), freq='min')  
time_index = pd.DatetimeIndex(time_index)
df = df.set_index(time_index)
df = df.drop(['time'], axis=1)

print(df.shape, '\n', df.columns)

########################## Modeling ##########################
# Task: Predicting future energy consumption by utilizing weather information
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
num_modalities = 10
corr_threshold = -1.0
use_independent = True
use_dependent = False
interaction_model = 'mlp' # 'linear' MLPRegressor or LinearRegression
criterion = mean_squared_error

use_ica = True  # default: True (for baseline comparison)
if not use_ica:
    use_independent = True
    use_dependent = False

# Define datasets and labels TODO: transform icon and summary to one-hot
df_weather = df[['temperature',
    #    'icon', 
       'humidity', 'visibility', 
    #    'summary', 
       'apparentTemperature',
       'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity',
       'dewPoint', 'precipProbability', 'use']]
print(df_weather.shape, '\n', df_weather.head(), print(df_weather.dtypes))

# Drop dulicated rows
df_weather = df_weather.drop_duplicates()

# Select modalities
df_weather = df_weather.iloc[:, [i for i in range(num_modalities)] + [-1]]

# Normalization
normalization_method = 'normal'
df_weather = normalization(df_weather, normalization_method, data_type='df') 
print(sum(df_weather.iloc[:, 0]), max(df_weather.iloc[:, 0]), min(df_weather.iloc[:, 0]))
print(df_weather.iloc[:, 0].mean(), df_weather.iloc[:, 0].std)

# Train test split
df_train = df_weather.sample(frac = 0.8, random_state=seed)
df_test = df_weather.drop(df_train.index)
print(df_weather.shape, df_train.shape, df_test.shape)

y_train = df_train[['use']].values.ravel()
X_train = df_train.drop(columns=['use'])
y_test = df_test[['use']].values.ravel()
X_test = df_test.drop(columns=['use'])

# CC-ICA
if use_ica:
    num_buckets = 4
    oracle_test_transfrom = True
    same_cc_test = True
    
    df_train["buckets"] = pd.qcut(df_train["use"], q=[i/num_buckets for i in range(num_buckets)] + [1], labels=[i for i in range(num_buckets)])
    df_test["buckets"] = pd.qcut(df_test["use"], q=[i/num_buckets for i in range(num_buckets)] + [1], labels=[i for i in range(num_buckets)])
    ica_X_train = X_train.copy()
    ica_X_test = X_test.copy()
    
    ica_list = []
    for i in range(num_buckets):
        cc_X_train = X_train.loc[df_train["buckets"]==i]
        ICA = FastICA(n_components=num_modalities, random_state=0, whiten='unit-variance')
        cc_ica_X_train = ICA.fit_transform(cc_X_train)
        ica_X_train.loc[cc_X_train.index] = cc_ica_X_train
        if same_cc_test:
            cc_X_test = X_test.loc[df_test["buckets"]==i]
            cc_ica_X_test = ICA.transform(cc_X_test)
            ica_X_test.loc[cc_X_test.index] = cc_ica_X_test

        ica_list.append(ICA)
        print(cc_X_train.shape, X_train.shape, ica_X_train.shape)
    print(ica_X_train.head(), '\n', X_train.head())

    # # TODO: how to transform X_test in cc-ica?
    # ica_X_test = ICA.transform(X_test)
    # print(np.mean(ica_X_train.values[:, 0], axis=0), np.std(ica_X_train.values[:, 0], axis=0))

# check if the class ica components are different
for ica in ica_list:
    print(ica.components_[:3, :num_buckets])

ica_X_train = np.array(ica_X_train)
ica_X_test = np.array(ica_X_test)

# LASSO baseline
reg = LassoLars(alpha=1., normalize=False)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
test_score = criterion(y_pred, y_test)
print(f"**(LASSO)** mse: {test_score:.4f}")

reg = LassoLars(alpha=1., normalize=False)
reg.fit(ica_X_train, y_train)
y_pred = reg.predict(ica_X_test)
test_score = criterion(y_pred, y_test)
print(f"**(LASSO | Independent)** mse: {test_score:.4f}")

# dependent components
dependent_X_train = np.array(X_train) - ica_X_train
dependent_X_test = np.array(X_test) - ica_X_test

# Compute correlations
intersections = get_intersections(num_modalities=num_modalities)
uni_modal_index = [inter for inter in intersections if len(inter)==1 ]
bi_intersections = [inter for inter in intersections if len(inter)==2 ]
remain_modalities = [i for i in range(num_modalities)]

# Train
print(f"Start Training...")
model_list = []
test_score_list = []
corr_list = []

# First model
corrs_ = [get_corr(ica_X_train[:, i], y_train) for i in remain_modalities]
corrs = [abs(x) for x in corrs_]
modality_index = corrs.index(max(corrs))
remain_modalities.remove(modality_index)

if not max(corrs) < corr_threshold:
    corr_list.append(max(corrs))
    
    print(f'Training unimodal model for modality {modality_index}..')
    model = LinearRegression().fit(ica_X_train[:, modality_index:modality_index+1], y_train)
    y_pred = model.predict(ica_X_test[:, modality_index:modality_index+1])
    test_score = criterion(y_pred, y_test)
    print(f"**(unimodal)** Modality: {modality_index}, corr: {corrs_[corrs.index(max(corrs))]:.4f}, mse: {test_score:.4f}")
    model_list.append({'model': model, 'modality':[modality_index]})
    test_score_list.append(test_score)

# Residual uni-model
while remain_modalities:
    y_train_res = get_res_y(model_list, ica_X_train, y_train)

    corrs_ = [get_corr(ica_X_train[:, i], y_train_res) for i in remain_modalities]
    corrs = [abs(x) for x in corrs_]
    modality_index = remain_modalities[corrs.index(max(corrs))]
    remain_modalities.remove(modality_index)

    if not max(corrs) < corr_threshold:
        corr_list.append(max(corrs))
    
        model = LinearRegression().fit(ica_X_train[:, [modality_index]], y_train_res)
        y_pred = model.predict(ica_X_test[:, [modality_index]])
        y_res = add_res_y(model_list, ica_X_test)
        test_score = criterion(y_pred+y_res, y_test)
        print(f"**(unimodal)** Modality: {modality_index}, corr: {corrs_[corrs.index(max(corrs))]:.4f}, test score: {test_score:.4f}")
        
        model_list.append({'model': model, 'modality':[modality_index]})
        test_score_list.append(test_score)

# Residual bi-model (independent)
if use_independent:
    while bi_intersections:
        y_train_res = get_res_y(model_list, ica_X_train, y_train)
        # NOTE: checked the two functions are the same

        corrs_ = []
        for inter in bi_intersections:
            modality_index = [int(inter[0])-1, int(inter[1])-1]
            X_train_ = np.multiply(ica_X_train[:, modality_index[0]], ica_X_train[:, modality_index[1]])
            corrs_.append(get_corr(X_train_, y_train_res))

        corrs = [abs(x) for x in corrs_]
        inter = bi_intersections[corrs.index(max(corrs))]
        modality_index = [int(inter[0])-1, int(inter[1])-1]
        bi_intersections.remove(inter)

        if not max(corrs) < corr_threshold:
            corr_list.append(max(corrs))
        
            # X_train_ = np.multiply(ica_X_train[:, [modality_index[0]]], ica_X_train[:, [modality_index[1]]])
            X_train_ = ica_X_train[:, modality_index]
            if interaction_model == 'mlp':
                model = MLPRegressor(random_state=1, max_iter=500).fit(X_train_, y_train_res)
            elif interaction_model == 'linear':
                model = LinearRegression().fit(X_train_, y_train_res)

            # X_test_ = np.multiply(ica_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]])
            X_test_ = ica_X_test[:, modality_index]
            y_pred = model.predict(X_test_)
            y_res = add_res_y(model_list, ica_X_test)
            test_score = criterion(y_pred+y_res, y_test)
            print(f"**(independent)** Modality: {modality_index}, corr: {corrs_[corrs.index(max(corrs))]:.4f}, test score: {test_score:.4f}")
            
            model_list.append({'model': model, 'modality':modality_index})
            test_score_list.append(test_score)

# Residual bi-model (dependent)
if use_dependent:
    bi_intersections = [inter for inter in intersections if len(inter)==2 ]
    y_train_res_old = get_res_y(model_list, ica_X_train, y_train)
    y_test_res_old = add_res_y(model_list, ica_X_test)

    model_list = []
    while bi_intersections:
        y_train_res = get_res_y_2(model_list, dependent_X_train, ica_X_train, y_train_res_old) if len(model_list) else y_train_res_old

        corrs_ = []
        for inter in bi_intersections:
            modality_index = [int(inter[0])-1, int(inter[1])-1]
            X_train_ = np.multiply(dependent_X_train[:, modality_index[0]], ica_X_train[:, modality_index[1]])
            corrs_.append(get_corr(X_train_, y_train_res))
        corrs = [abs(x) for x in corrs_]
        inter = bi_intersections[corrs.index(max(corrs))]
        modality_index = [int(inter[0])-1, int(inter[1])-1]
        bi_intersections.remove(inter)

        if not max(corrs) < corr_threshold:
            corr_list.append(max(corrs))
        
            # X_train_ = np.multiply(ica_X_train[:, [modality_index[0]]], ica_X_train[:, [modality_index[1]]])
            X_train_ = np.concatenate((dependent_X_train[:, [modality_index[0]]], ica_X_train[:, [modality_index[1]]]), axis=1)
            if interaction_model == 'mlp':
                model = MLPRegressor(random_state=1, max_iter=500).fit(X_train_, y_train_res)
            elif interaction_model == 'linear':
                model = LinearRegression().fit(X_train_, y_train_res)

            # X_test_ = np.multiply(ica_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]])
            X_test_ = np.concatenate((dependent_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]]), axis=1)
            y_pred = model.predict(X_test_)
            y_res = add_res_y_2(model_list, dependent_X_test, ica_X_test) if len(model_list) else np.zeros_like(y_test)
            test_score = criterion(y_pred+y_res+y_test_res_old, y_test)
            print(f"**(dependent)** Modality: {modality_index}, corr: {corrs_[corrs.index(max(corrs))]:.4f}, test score: {test_score:.4f}")
            
            model_list.append({'model': model, 'modality':modality_index})
            test_score_list.append(test_score)

print('corr threshold: ', corr_threshold)
print('test acc: ', test_score_list)
print('abs corr: ', corr_list)

plt.figure(0)
plt.plot([x for x in range(len(test_score_list))], test_score_list)
plt.savefig('score.png')