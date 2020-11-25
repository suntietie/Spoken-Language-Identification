import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import Config, plot_epoch


### MODEL
class Model(nn.Module):
    '''Model structure
    :input: (batch_size, len_seq, num_feature)
    
    => RNN hidden_state: (2, batch_size, hidden_size)
    => Linear(softmax)
    
    :output: (batch_size, 1, output_size=3)
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ## GRU with two layer of hidden layer
        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=4, dropout=0.3, batch_first=True).to(self.device)
        ## full connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
       
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size)
    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(self.device)
        out, h_out = self.rnn(x, h)
        h_out = h_out.to(self.device)
        out = self.fc(out)
        out = out.to(self.device)
        
        #return out, h
        return out
        

### MODEL
num_feature = 64
hidden_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gru_model = Model(input_size=num_feature, hidden_size=hidden_size, output_size=3).to(device)

