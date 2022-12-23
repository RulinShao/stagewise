from sythetic import *
import pandas as  pd
from sklearn.decomposition import FastICA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from numpy import cov
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def calculate_accuracy(y_pred, y):
  pred = (y_pred >= 0.5).float()
  acc = (pred == y).sum() / y.shape[0]
  return acc

def get_corr(X, y):
    corr, _ = pearsonr(X, y)
    return corr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
num_modalities = 4
num_data = batch_size = 10000
corr_threshold = -1.0
use_independent = True
use_dependent = False

# Data
feature_dim_info = dict()
label_dim_info = dict()

intersections = get_intersections(num_modalities=num_modalities)
for x in intersections:
    feature_dim_info[x] = 10
    label_dim_info[x] = 8

# Get dataset
print(f"Generating dataset ...")
total_data, total_labels = generate_data(num_data, num_modalities, feature_dim_info, label_dim_info)
dataset = MultimodalDataset(total_data, total_labels)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*num_data), num_data-int(0.8*num_data)])
# print(test_dataset)
train_dataset_pd = pd.DataFrame(torch.Tensor(train_dataset).numpy())
X_train = train_dataset_pd.iloc[:, [i for i in range(num_modalities)]]
y_train = train_dataset_pd.iloc[:, -1]
# print(y_train.shape, X_train.shape)

test_dataset_pd = pd.DataFrame(torch.Tensor(test_dataset).numpy())
X_test = test_dataset_pd.iloc[:, [i for i in range(num_modalities)]]
y_test = test_dataset_pd.iloc[:, -1]

ICA = FastICA(n_components=num_modalities, random_state=0, whiten='unit-variance')
ica_X_train = ICA.fit_transform(X_train)
ica_X_test = ICA.transform(X_test)


# get dependent components
dependent_X_train = np.array(X_train) - ica_X_train
dependent_X_test = np.array(X_test) - ica_X_test
corr_matrix = np.zeros((num_modalities, num_modalities))
# print(dependent_X_train.shape, ica_X_train.shape)
for i in range(num_modalities):
    for j in range(num_modalities):
        corr_matrix[i][j] = get_corr(dependent_X_train[:, i], np.array(X_train)[:, j])
plt.matshow(corr_matrix)
plt.colorbar()
plt.savefig('corr.jpg')
plt.plot()

# TODO: polynomial dependent features
# poly_model = PolynomialFeatures(degree=degree)
# poly_x_values = poly_model.fit_transform(x_values)

# train_dataset = TensorDataset(torch.tensor(ica_X_train), torch.tensor(y_train))
# test_dataset = TensorDataset(torch.tensor(ica_X_test), torch.tensor(y_test))
# train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False,
#                             batch_size=batch_size,
#                             num_workers=4)
# test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
#                             batch_size=batch_size,
#                             num_workers=4)

# Compute correlations
uni_modal_index = [inter for inter in intersections if len(inter)==1 ]
bi_intersections = [inter for inter in intersections if len(inter)==2 ]
remain_modalities = [i for i in range(num_modalities)]

print(f"Start Training...")
model_list = []
test_acc_list = []
corr_list = []

# # LARS
# reg = LassoLars(alpha=.1, normalize=False)
# reg.fit(ica_X_train, y_train)
# y_pred = reg.predict(ica_X_test)
# test_acc = calculate_accuracy(torch.tensor(y_pred), torch.tensor(y_test))
# print(f"acc: {test_acc:.4f}")

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import linear_model

# print("Computing regularization path using the LARS ...")
# # _, _, coefs = linear_model.lars_path(X_train.values, y_train, method="lars", verbose=True)
# _, _, coefs = linear_model.lars_path(ica_X_train, y_train, method="lars", verbose=True)

# xx = np.sum(np.abs(coefs.T), axis=1)
# xx /= xx[-1]

# plt.plot(xx, coefs.T)
# ymin, ymax = plt.ylim()
# plt.vlines(xx, ymin, ymax, linestyle="dashed")
# plt.xlabel("|coef| / max|coef|")
# plt.ylabel("Coefficients")
# plt.title("LASSO Path")
# plt.axis("tight")
# plt.savefig('lars_path_ica.png')
# stop

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
    test_acc = calculate_accuracy(torch.tensor(y_pred), torch.tensor(y_test))
    print(f"Modality: {modality_index}, acc: {test_acc:.4f}, corr: {corrs_[corrs.index(max(corrs))]:.4f}")
    model_list.append({'model': model, 'modality':[modality_index]})
    test_acc_list.append(test_acc)

def get_res_y(model_list, X, y):
    for model in model_list:
        modality_index = model['modality']
        if len(modality_index) == 1:
            X_train_ = X[:, modality_index]
        else:
            X_train_ = X[:, modality_index]
            # X_train_ = np.multiply(X[:, [modality_index[0]]], X[:, [modality_index[1]]])
        y_pred = model['model'].predict(X_train_)
        y = y - y_pred
    return y

def get_res_y_2(model_list, X_1, X_2, y):
    for model in model_list:
        modality_index = model['modality']
        X_train_ = np.concatenate((X_1[:, [modality_index[0]]], X_2[:, [modality_index[1]]]), axis=1)
        y_pred = model['model'].predict(X_train_)
        y = y - y_pred
    return y

def add_res_y(model_list, X):
    for i, model in enumerate(model_list):
        modality_index = model['modality']
        if len(modality_index) == 1:
            X_train_ = X[:, modality_index]
        else:
            X_train_ = X[:, modality_index]
            # X_train_ = np.multiply(X[:, [modality_index[0]]], X[:, [modality_index[1]]])
        
        if i == 0:   
            y = model['model'].predict(X_train_)
        else:
            y = y + model['model'].predict(X_train_)
    return y

def add_res_y_2(model_list, X_1, X_2):
    for i, model in enumerate(model_list):
        modality_index = model['modality']
        X_train_ = np.concatenate((X_1[:, [modality_index[0]]], X_2[:, [modality_index[1]]]), axis=1)
        
        if i == 0:   
            y = model['model'].predict(X_train_)
        else:
            y = y + model['model'].predict(X_train_)
    return y

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
        test_acc = calculate_accuracy(torch.tensor(y_pred+y_res), torch.tensor(y_test))
        print(f"**(unimodal)** Modality: {modality_index}, acc: {test_acc:.4f}, corr: {corrs_[corrs.index(max(corrs))]:.4f}")
        
        model_list.append({'model': model, 'modality':[modality_index]})
        test_acc_list.append(test_acc)

# Residual bi-model (independent)
if use_independent:
    while bi_intersections:
        y_train_res = get_res_y(model_list, dependent_X_train, y_train)

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
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(random_state=1, max_iter=500).fit(X_train_, y_train_res)
            # model = LinearRegression().fit(X_train_, y_train_res)

            # X_test_ = np.multiply(ica_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]])
            X_test_ = ica_X_test[:, modality_index]
            y_pred = model.predict(X_test_)
            y_res = add_res_y(model_list, ica_X_test)
            test_acc = calculate_accuracy(torch.tensor(y_pred+y_res), torch.tensor(y_test))
            print(f"**(independent)** Modality: {modality_index}, acc: {test_acc:.4f}, corr: {corrs_[corrs.index(max(corrs))]:.4f}")
            
            model_list.append({'model': model, 'modality':modality_index})
            test_acc_list.append(test_acc)

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
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(random_state=1, max_iter=500).fit(X_train_, y_train_res)
            # model = LinearRegression().fit(X_train_, y_train_res)

            # X_test_ = np.multiply(ica_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]])
            X_test_ = np.concatenate((dependent_X_test[:, [modality_index[0]]], ica_X_test[:, [modality_index[1]]]), axis=1)
            y_pred = model.predict(X_test_)
            y_res = add_res_y_2(model_list, dependent_X_test, ica_X_test) if len(model_list) else np.zeros_like(y_test)
            test_acc = calculate_accuracy(torch.tensor(y_pred+y_res+y_test_res_old), torch.tensor(y_test))
            print(f"**(dependent)** Modality: {modality_index}, acc: {test_acc:.4f}, corr: {corrs_[corrs.index(max(corrs))]:.4f}")
            
            model_list.append({'model': model, 'modality':modality_index})
            test_acc_list.append(test_acc)

print('corr threshold: ', corr_threshold)
print('test acc: ', test_acc_list)
print('abs corr: ', corr_list)

plt.figure(0)
plt.plot([x for x in range(len(test_acc_list))], test_acc_list)
plt.savefig('acc.png')