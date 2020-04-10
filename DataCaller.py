import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataCaller(Dataset):
    def __init__(self, data_path, label_path, input_size):
        
        self.data_path = data_path
        self.label_path = label_path
        self.input_size = input_size
        label = pd.read_excel(open(self.label_path, 'rb'), sheet_name='Sheet2', header=None)
        self.label = torch.from_numpy( np.array(label) ).float()

        self.data = []
        for f in os.listdir(self.data_path):
            path = os.path.join(self.data_path, f)
            img = cv2.imread(path)
            img = cv2.resize(img, (self.input_size, self.input_size))
            self.data.append(np.array(img))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y