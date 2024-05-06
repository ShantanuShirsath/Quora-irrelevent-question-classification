import pandas as pd
import sys
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from data_transformer import InitiateDataTransformation
import torch
import os
import dataloader
import RNN_model
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from RNN_model import SimpleRNN
import torch.optim as optim
import torch.nn as nn
from train_test import train_test_module
import pickle

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    val_data_path = os.path.join('artifacts',"val.csv")

class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()
            self.train_data = pd.read_csv("data/train.csv")
        except Exception as e:
            raise CustomException(e,sys)

    def data_spliting(self):
        try:
            df = self.train_data
            # Splitting the data into training and test sets
            train_set_1,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            # Splitting the training data into training and validation sets
            train_set, validation_set=train_test_split(train_set_1,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            validation_set.to_csv(self.ingestion_config.val_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.val_data_path,
                    self.ingestion_config.test_data_path
                    )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, val_data, test_data = obj.data_spliting()
    print("--> done data splitting")
    # Initialize DataTransformation class and perform data transformation
    data_transformation = InitiateDataTransformation()
    print("--> data transformation object created")
    x_train, x_val, x_test,vocabulary_size,_=data_transformation.initiate_data_transformation(train_data, val_data, test_data)
    # Load labels for training and validation data
    print("--> data transformation")
    y_1 = pd.read_csv("artifacts/train.csv")
    y_train = torch.LongTensor(y_1['target'])
    y_2 = pd.read_csv("artifacts/val.csv")
    y_val = torch.LongTensor(y_2['target'])
    # creating Dataloaders
    batch_size = 64
    input_size = vocabulary_size 
    hidden_size = 128
    output_size = 2
    load_data = dataloader.LoadData(batch_size=batch_size)
    train_loader = load_data.create_dataloader(x_train,y_train)
    val_loader = load_data.create_dataloader(x_val,y_val)
    print("--> data loader created")
    model = SimpleRNN(input_size, hidden_size, output_size)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    traintest = train_test_module(trainloader = train_loader,
                                   valloader= val_loader,
                                   device = torch.device("cuda"),
                                   optimizer=optimizer,
                                   net = model,
                                   criterion= criterion,
                                   batch_size= batch_size)
    print("--> strating training:")
    epoch = 20
    epoch_loss = []*epoch 
    epoch_Accuracy = []*epoch
    epoch_val_loss = []*epoch 
    epoch_val_Accuracy = []*epoch
    
    for epoch_index in range(epoch):
        print(f'Epoch: {epoch_index}\n')

        train_loss, train_accuracy = traintest.train_one_epoch()
        val_loss, val_accuracy = traintest.validate_one_epoch()

        # Append the values to the lists
        epoch_loss.append(train_loss)
        epoch_Accuracy.append(train_accuracy)

        # Optionally, you can also append validation values
        epoch_val_loss.append(val_loss)
        epoch_val_Accuracy.append(val_accuracy)

    print('Finished Training')

    # Define the file path to save the model
    model_path = 'artifacts/model.pkl'
    # Move the model to CPU
    model_cpu = model.to('cpu')
    # Serialize and save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model_cpu, f)

    y_3 = pd.read_csv("artifacts/test.csv")
    y_test = torch.LongTensor(y_3['target'])

    test_loader = load_data.create_dataloader(x_test,y_test)
    test_accuracy, test_f1 = traintest.test_one_epoch(test_loader)

    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test F1 Score: {test_f1:.4f}")


