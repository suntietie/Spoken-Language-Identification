import numpy as np
import os
import os.path as osp
import argparse
import matplotlib.pyplot as plt
Config ={}

## if in PC environment
Config['root_path'] = '/Users/tieming/code/dataset/train'
## if in AWS environment
# Config['root_path] = '/mnt/train'

Config['english'] = osp.join(Config['root_path'], 'train_english')
Config['hindi'] = osp.join(Config['root_path'], 'train_hindi')
Config['mandarin'] = osp.join(Config['root_path'], 'train_mandarin')
Config['mfcc_feature'] = 64

# features after MFCC preprocessing
Config['hw5_data.hdf5'] = osp.join(Config['root_path'], "hw5_data.hdf5")

Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 5
Config['batch_size'] = 32

Config['learning_rate'] = 0.002
Config['num_workers'] = 5
Config['num_points_plot'] = 5


def plot_epoch(x_list, y_list, fname, num_epochs=Config['num_epochs'], accuracy="Accuracy"):
    l = [i for i in range(1, len(x_list)+1)]
    new_ticks=np.linspace(0,num_epochs,Config['num_points_plot'])
    plt.plot(l, x_list,label="Training set")
    plt.plot(l, y_list,label="Validation set")

    plt.xticks(new_ticks)
    plt.title("{accuracy} Performance Versus Epoch")
    plt.legend(labels=["Training set", "Validation set"],loc='best')
    plt.xlabel("Epoches")
    plt.ylabel(accuracy)
    plt.savefig(fname=fname)
    plt.close()
    return 