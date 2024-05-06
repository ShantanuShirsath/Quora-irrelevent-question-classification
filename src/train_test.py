import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import f1_score


class train_test_module(nn.Module):
    def __init__(self,trainloader,valloader,device,optimizer,net,criterion,batch_size):
        super(train_test_module,self).__init__()
        self.trainloader = trainloader
        self.device = device
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.batch_size = batch_size
        self.valloader = valloader      
    
    def train_one_epoch(self):
        try:

            self.net.train(True)
            self.net.to(self.device)

            running_loss = 0.0
            running_accuracy = 0.0
            Epoch_loss = []
            Epoch_accuracy = []
            
            for batch_index, data in enumerate(self.trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(inputs)
                correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                running_accuracy += correct / self.batch_size

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if batch_index % 5000 == 4999:
                    avg_loss_across_batches = running_loss / 5000
                    avg_acc_across_batches = (running_accuracy / 5000) * 100
                    print('Batch {0}, Loss: {1:.5f}, Accuracy: {2:.3f}%'.format(batch_index+1,
                                                                        avg_loss_across_batches,
                                                                        avg_acc_across_batches))
                    running_loss = 0.0
                    running_accuracy = 0.0
                    Epoch_loss.append(avg_loss_across_batches)
                    Epoch_accuracy.append(avg_acc_across_batches)

            print()
            
            return  sum(Epoch_loss)/len(Epoch_loss) , sum(Epoch_accuracy)/len(Epoch_accuracy)
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def validate_one_epoch(self):

        try:

            self.net.train(False)
            running_loss = 0.0
            running_accuracy = 0.0
            Epoch_loss = []
            Epoch_accuracy = []

            for i, data in enumerate(self.valloader):
                input, labels = data[0].to(self.device), data[1].to(self.device)

                with torch.no_grad():
                    outputs = self.net(input)
                    correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                    running_accuracy += correct / self.batch_size
                    loss = self.criterion(outputs, labels) # One number, the average batch loss
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(self.valloader)
            avg_acc_across_batches = (running_accuracy / len(self.valloader)) * 100

            print('Val Loss: {0:.5f}, Val Accuracy: {1:.3f}%'.format(avg_loss_across_batches,
                                                                    avg_acc_across_batches))
            print('***************************************************')
            print()
            Epoch_loss.append(avg_loss_across_batches)
            Epoch_accuracy.append(avg_acc_across_batches)
            
            return  Epoch_loss,Epoch_accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def test_one_epoch(self,testloader):
        try:

            self.net.eval()  # Set the model to evaluation mode
            self.net.to(self.device)
            test_loss = 0.0
            correct = 0
            total = 0
            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to device
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predicted_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
            accuracy = correct / total
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            return accuracy, f1
        
        except Exception as e:
            raise CustomException(e,sys)