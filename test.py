import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import Config, plot_epoch
from model import gru_model


model = gru_model
model.fc.num_feature