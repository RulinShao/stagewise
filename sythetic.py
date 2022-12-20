import torch
import numpy as np
import math

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultimodalDataset(Dataset):
  def __init__(self, total_data, total_labels):
    self.data = torch.from_numpy(total_data).float()
    self.labels = torch.from_numpy(total_labels)
    self.num_modalities = self.data.shape[0]
  
  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return tuple([self.data[i, idx] for i in range(self.num_modalities)] + [self.labels[idx]])


def get_intersections(num_modalities):
  modalities = [i for i in range(1, num_modalities+1)]
  all_intersections = [[]]
  for i in modalities:
    new = [s + [str(i)] for s in all_intersections]
    all_intersections += new
  res = list(map(lambda x: ''.join(x), sorted(all_intersections[1:])))
  return sorted(res, key=lambda x: (len(x), x))


def generate_data(num_data, num_modalities, feature_dim_info, label_dim_info, transforms=None):
  # Standard deviation of generated Gaussian distributions
  SEP = 0.5
  default_transform_dim = 1

  total_data = [[] for i in range(num_modalities)]
  total_labels = []


  # define transform matrices if not provided
  modality_dims = [0]*num_modalities
  for i in range(1, num_modalities+1):
      for k, d in feature_dim_info.items():
        if str(i) in k:
          modality_dims[i-1] += d

  if transforms is None:
      transforms = []
      for i in range(num_modalities):
        transforms.append(np.random.uniform(0.0,1.0,(modality_dims[i], default_transform_dim)))


  # generate data
  for data_idx in range(num_data):

    # get Gaussian data vector for each modality
    raw_features = dict()
    for k, d in feature_dim_info.items():
      raw_features[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d)*0.5, (1,))[0]

    
    modality_concept_means = []
    for i in range(1, num_modalities+1):
      modality_concept_means.append([])
      for k, v in raw_features.items():
        if str(i) in k:
          modality_concept_means[-1].append(v)

    raw_data = [np.concatenate(modality_concept_means[i]) for i in range(num_modalities)]
    

    # Transform into high-dimensional space
    modality_data = [raw_data[i] @ transforms[i] for i in range(num_modalities)]


    # update total data
    for i in range(num_modalities):
      total_data[i].append(modality_data[i])

    # get label vector
    label_components = []
    for k,d in label_dim_info.items():
      label_components.append(raw_features[k][:d])
   
    label_vector = np.concatenate(label_components + [np.random.randint(0, 2, 1)]) 
    label_prob = 1 / (1 + math.exp(-np.mean(label_vector)))
    total_labels.append([int(label_prob >= 0.5)])

      
  total_data = np.array(total_data)
  total_labels = np.array(total_labels)

  return total_data, total_labels


def get_data_loaders(num_modalities, num_data, batch_size):
  # Define custom dimensions of features and labels
  feature_dim_info = dict()
  label_dim_info = dict()

  intersections = get_intersections(num_modalities=num_modalities)
  for x in intersections:
    feature_dim_info[x] = 10
    label_dim_info[x] = 4

  print(intersections)
  print(feature_dim_info)
  print(label_dim_info)

  # Get dataset
  total_data, total_labels = generate_data(num_data, num_modalities, feature_dim_info, label_dim_info)
  dataset = MultimodalDataset(total_data, total_labels)
  print(total_data.shape, total_labels.shape)

  # Dataloader
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*num_data), num_data-int(0.8*num_data)])

  train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                            batch_size=batch_size,
                            num_workers=4)
  test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                            batch_size=batch_size,
                            num_workers=4)
  return train_loader, test_loader


class UnimodalModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, res=False):
    super().__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.clf_head = nn.Linear(hidden_dim, output_dim)
    self.res = res
  
  def forward(self, x):
    h = F.relu(self.layer1(x))
    if not self.res:
      out = torch.sigmoid(self.clf_head(h))
    else:
      out = torch.tanh(self.clf_head(h))
    return out


class BimodalModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, res=False):
    super().__init__()
    self.layer1 = nn.Linear(input_dim*2, hidden_dim)
    self.clf_head = nn.Linear(hidden_dim, output_dim)
    self.res = res
  
  def forward(self, x):
    h = F.relu(self.layer1(x))
    if not self.res:
      out = torch.sigmoid(self.clf_head(h))
    else:
      out = torch.tanh(self.clf_head(h))
    return out
  
  
def calculate_accuracy(y_pred, y):
  pred = (y_pred >= 0.5).float()
  acc = (pred == y).sum() / y.shape[0]
  return acc

def train_uni(model, iterator, optimizer, criterion, modality_index):
  modality_index = modality_index[0]
  epoch_loss = 0
  epoch_acc = 0

  model.train()

  for i_batch, data_batch in enumerate(iterator):
      x = data_batch[:-1][modality_index].to(device)
      y = data_batch[-1].float().to(device)

      optimizer.zero_grad()
      y_pred = model(x)

      loss = criterion(y_pred, y)
      acc = calculate_accuracy(y_pred, y)

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_uni(model, iterator, criterion, modality_index):
  modality_index = modality_index[0]
  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for i_batch, data_batch in enumerate(iterator):
      x = data_batch[:-1][modality_index].to(device)
      y = data_batch[-1].float().to(device)

      y_pred = model(x)

      loss = criterion(y_pred, y)
      acc = calculate_accuracy(y_pred, y)

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_res(model, iterator, optimizer, criterion, modality_index, model_list):
  epoch_loss = 0
  epoch_acc = 0

  model.train()
  for model_res in model_list:
    model_res['model'].eval()

  for i_batch, data_batch in enumerate(iterator):
      data = data_batch[:-1]
      if len(modality_index) == 1:
        x = data[modality_index[0]].to(device)
      elif len(modality_index) == 2:
        x = torch.cat([data[modality_index[0]], data[modality_index[1]]], dim=-1).to(device)
      y = data_batch[-1].float().to(device)

      x_res_list = []
      for j in range(len(model_list)):
        modality = model_list[j]['modality']
        if len(modality) == 1:
          x_res_list.append(data[modality[0]].to(device))
        elif len(modality) == 2:
          x_res_list.append(torch.cat([data[modality[0]], data[modality[1]]], dim=-1).to(device))
      
      optimizer.zero_grad()
      y_pred = model(x)

      
      y_res = y
      additive_pred = y_pred.detach().clone()
      for j in range(len(model_list)):
        model_res = model_list[j]['model']
        with torch.no_grad():
          y_pred_res = model_res(x_res_list[j])
        y_res = y_res - y_pred_res    # y_pred --> res = y - y_pred_res
        additive_pred += y_pred_res

      loss = criterion(y_pred, y_res)
      acc = calculate_accuracy(additive_pred, y)

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval_res(model, iterator, criterion, modality_index, model_list):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()
  for model_res in model_list:
    model_res['model'].eval()

  with torch.no_grad():
    for i_batch, data_batch in enumerate(iterator):
      data = data_batch[:-1]
      if len(modality_index) == 1:
        x = data[modality_index[0]].to(device)
      elif len(modality_index) == 2:
        x = torch.cat([data[modality_index[0]], data[modality_index[1]]], dim=-1).to(device)
      y = data_batch[-1].float().to(device)

      x_res_list = []
      for j in range(len(model_list)):
        modality = model_list[j]['modality']
        if len(modality) == 1:
          x_res_list.append(data[modality[0]].to(device))
        elif len(modality) == 2:
          x_res_list.append(torch.cat([data[modality[0]], data[modality[1]]], dim=-1).to(device))

      y_pred = model(x)
      y_res = y
      additive_pred = y_pred.detach().clone()
      for j in range(len(model_list)):
        model_res = model_list[j]['model']
        y_pred_res = model_res(x_res_list[j])
        y_res = y_res - y_pred_res    # y_pred --> res = y - y_pred_res
        additive_pred += y_pred_res

      loss = criterion(y_pred, y_res)
      acc = calculate_accuracy(additive_pred, y)

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Hyperparameters
  A_dim, B_dim = 100, 100
  label_dim = 1
  lr = 1e-4
  hidden_dim=512 
  embed_dim=128
  layers=1
  activation = 'relu'

  # Data
  num_modalities = 3
  num_data = 10000
  batch_size = 512
  train_loader, test_loader = get_data_loaders(num_modalities, num_data, batch_size)

  EPOCHS = 100

  model_list = []
  max_test_acc = []
  for modality_index in range(num_modalities):
    
    model = UnimodalModel(100, 64, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f'Training unimodal model for modality {modality_index}..')
    max_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_uni(model, train_loader, optimizer, criterion, modality_index)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

        test_loss, test_acc = eval_uni(model, test_loader, criterion, modality_index)    
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
        if max_acc < test_acc:
          max_acc = test_acc
    
    max_test_acc.append(max_acc)
    model_list.append(model)

  for modality_idx in range(num_modalities):
    test_loss, test_acc = eval_uni(model_list[modality_idx], test_loader, criterion)   
    print(f'\t Modality: {modality_idx} | Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}% | Best. Acc: {max_test_acc[modality_idx]*100:.2f}%')