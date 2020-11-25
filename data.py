import numpy as np
import os
import os.path as osp
import argparse
from tqdm import tqdm
import librosa
from utils import Config
import h5py


def mel_features(filename, thres=40):
    '''
    @param thres: The threshold (in decibels) below reference to consider as silence
    @return mfccs: np.ndarray, shape=(num_features,n_mfcc)
    '''
    y, sr = librosa.load(filename, sr=16000)
    y = librosa.core.to_mono(y)
    intervals = librosa.effects.split(y, top_db=thres)
    
    y_cons = np.zeros((0,))
    for interval in intervals:
        y_cons = np.hstack([ y_cons, y[interval[0]: interval[1]] ])
        
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length= int ( sr *0.010) )
    return mfccs.T



def processing(language_list, language="mandarin"):
    language_mfcc = []
    for filename in tqdm(language_list):
        mfccs = mel_features(osp.join(Config[language], filename), thres=40)    
        language_mfcc.append(mfccs)
    language_mfcc = np.array(language_mfcc)

    language_2D = np.concatenate([x for x in language_mfcc])
    N_lan = language_2D.shape[0] // sequence_length
    # make sure that I can utilize reshape
    language_2D = language_2D[:N_lan*sequence_length]
    language_3D = language_2D.reshape((N_lan, sequence_length, 64))
    
    # lan_label = np.full(shape=(N_lan,sequence_length,1),fill_value=mapping[language])
    # set one hot label (depreciated)
    lan_label = np.full(shape=(N_lan,sequence_length,3), fill_value=np.array([int(i == mapping[language]) for i in range(3)]))

    lan_all = np.concatenate([language_3D, lan_label],axis=2)
    return lan_all



mapping = {'english':0,'hindi':1,'mandarin':2}

# 64-dimensional MFCC features in this HW
num_feature = Config['mfcc_feature']

english_list = os.listdir(Config['english'])
hindi_list = os.listdir(Config['hindi'])
mandarin_list = os.listdir(Config['mandarin'])
sequence_length = 801
# separate train and test dataset
english_train = english_list[:9*len(english_list)//10]
english_test = english_list[9*len(english_list)//10:]
hindi_train = hindi_list[:9*len(hindi_list)//10]
hindi_test = hindi_list[9*len(hindi_list)//10:]
mandarin_train = mandarin_list[:9*len(mandarin_list)//10]
mandarin_test = mandarin_list[9*len(mandarin_list)//10:]


english_all = processing(language_list=english_train, language="english")
mandarin_all = processing(language_list=mandarin_train, language="mandarin")
hindi_all = processing(language_list=hindi_train, language="hindi")
X_all = np.concatenate([english_all, mandarin_all, hindi_all],axis=0)
X_all = np.random.shuffle(X_all)


train_label = X_all[:,:,-3:]
train_data = X_all[:,:,:64]


with h5py.File(osp.join(Config['root_path'], "hw5_data2.hdf5") , 'w') as hf:
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('train_label', data=train_label)
#     hf.create_dataset('english_test', data=english_test)
#     hf.create_dataset('mandarin_test',data=mandarin_test)
#     hf.create_dataset('english_test',data=english_test)





