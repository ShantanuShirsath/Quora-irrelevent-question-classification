import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from exception import CustomException
from logger import logging
import sys

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LoadData():
    def __init__(self,batch_size):
        super(LoadData,self).__init__()
        self.batch_size = batch_size
    
    def create_dataloader(self,X,y):
        try:
            dataset = TextDataset(X,y)
            loader = DataLoader(dataset, batch_size=self.batch_size
                                , shuffle=True)
            return loader
        
        except Exception as e:
            raise CustomException(e,sys)
