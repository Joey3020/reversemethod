
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset


class DataCaller(Dataset):
    def __init__(self, data_path, input_size, label_size):
        
        self.data_path = data_path
        self.input_size = input_size
        self.label_size = label_size

        self.data = np.array(pd.read_csv(self.data_path), dtype=np.float)

        self.X = torch.from_numpy(self.data[:, : -self.label_size ]).float()
        self.Y = torch.from_numpy(self.data[:, -self.label_size : ]).float()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.Y[idx])
        return x, y