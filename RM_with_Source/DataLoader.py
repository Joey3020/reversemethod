import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
class DataLoader():
    def __init__(self, batch_size, path):
        self.batch_size = batch_size
        self.path = path
        self.current_batch = 0
        # load data
        input1_pd = pd.read_excel(open(self.path, 'rb'), sheet_name = 'Sheet1', header = None)
        input_ten = torch.from_numpy( np.array(input1_pd) ).float()
        self.input = input_ten.reshape(-1,1,64,64)
        
        label = pd.read_excel(open(self.path, 'rb'), sheet_name = 'Sheet2', header = None)
        self.label = torch.from_numpy( np.array(label) ).float()
        
        self.data_size = len(label)
        print(self.data_size)
    
    def getbatchnum(self):
        return math.ceil(self.data_size / self.batch_size)
    
    def getbatch(self):
        current_batch = self.current_batch
        length = self.batch_size

        if current_batch + length < self.data_size :
            data_return = self.input[ current_batch : current_batch + length , :]
            label_return = self.label[current_batch : current_batch + length , :]
        else:
            data_return = self.input[current_batch : -1 , :]
            label_return = self.label[current_batch : -1 , :]
        
        self.current_batch += length
        if self.current_batch >= self.data_size:
            self.current_batch = 0
            
        return data_return, label_return
        
    def reset(self): #어디까지 리턴했는지 초기화
        self.current_batch = 0
    
    def makeimage(self, until):
        for idx in range(until):
            image = self.input[idx]
            image = image.numpy()
            print(image.shape)
            plt.imshow(image.reshape(64, 64), cmap="gray")
            plt.show()
            print(self.label[idx])