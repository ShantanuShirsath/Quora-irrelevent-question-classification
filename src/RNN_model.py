import torch
import torch.nn as nn
import torch.optim as optim
from exception import CustomException
from logger import logging
import sys

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # Take the hidden state from the last time step
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output
        
        