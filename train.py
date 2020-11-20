import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F

import numpy as np
import h5py
from utils import Config, plot_epoch
import os.path as osp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import time
import copy

class mfcc_set(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index:index+1]
        # remove dimension 0
        data = data.squeeze(0)
        label = self.y[index]
        return torch.Tensor(data), torch.Tensor(label)


# create Pytorch dataloader
def get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'], train_data=train_data,valid_data=valid_data,train_label=train_label,valid_label=valid_label,test_data=test_data,test_label=test_label):

    if debug == True:
        train_data, train_label = train_data[:100], train_label[:100]
        valid_data, valid_label = valid_data[:100], valid_label[:100]
        test_data, test_label = test_data[:100], test_label[:100]
    
    dataset_size = {'train': len(train_data), 'valid': len(valid_data), 'test': len(test_data)}

    train_set = mfcc_set(train_data, train_label)
    valid_set = mfcc_set(valid_data, valid_label)
    test_set = mfcc_set(test_data, test_label)

    datasets = {'train': train_set, 'valid': valid_set, 'test': test_set}

    dataloaders = {x: DataLoader(datasets[x],
                                shuffle=True if x=='train' else False,
                                batch_size=batch_size,
                                num_workers=num_workers)
                                for x in ['train', 'valid', 'test']}

    return dataset_size, dataloaders                            


def train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size):

    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):

                labels_temp = np.array(labels)
                labels = torch.Tensor(labels_temp.astype('long'))
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                pred = outputs
                pred[pred >= 0.5] = 1.0
                pred[pred <= 0.5] = 0.0
                running_corrects += torch.sum(pred==labels.data)


            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                acc_train_list.append(epoch_acc)
                loss_train_list.append(epoch_loss)
            if phase == 'valid':
                acc_test_list.append(epoch_acc)
                loss_test_list.append(epoch_loss)
            
            if phase=='valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # save model.pth - dictionary (best_model_wts)
        torch.save(best_model_wts, osp.join(Config['root_path'], 'mfcc_model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], 'mfcc_model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))



if __name__ == "__main__":

    with h5py.File(Config['hw5_data.hdf5'] , 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]

    train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.11, random_state=112)
    dataset_size, dataloaders = get_dataloader()

    ## use for plot
    acc_train_list,acc_test_list,loss_train_list,loss_test_list=[],[],[],[]

    ## change loss function
    criterion = nn.BCELoss()
    ## change optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'], weight_decay=0.0001)
    ## change device
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')   


    plot_epoch(acc_train_list, acc_test_list, "mfcc_acc.jpg", num_epochs=Config['num_epochs'])  
    plot_epoch(loss_train_list, loss_test_list, "mfcc_loss.jpg", num_epochs=Config['num_epochs'], accuracy="Loss") 