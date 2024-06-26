import pandas as pd
import torch
import dataloader
from logger import logging
from utils import load_object
from train_test import test_one_epoch
import torch.nn as nn
from exception import CustomException
import sys

def predict_test(df,batch_size):
    test_data = df
    preprocessor = load_object('artifacts\preprocessor.pkl')
    model = load_object('artifacts\model.pkl')
    load_data = dataloader.LoadData(batch_size=batch_size)
    X_test_preprocessed = preprocessor.get_data_transformation_object(test_data)
    print("--> data transformation")
    y_3 = pd.read_csv("artifacts/test.csv")
    y_test = torch.LongTensor(y_3['target'])
    device = "cuda"
    criterion = nn.CrossEntropyLoss()
    test_loader = load_data.create_dataloader(X_test_preprocessed,y_test)
    test_accuracy, test_f1 = test_one_epoch(testloader=test_loader, net = model, device = device, criterion = criterion)

    logging.info(f"Test Accuracy: {test_accuracy:.2%}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    
def predict_one_data(df):
    try:
        data = df
        preprocessor = load_object('artifacts\preprocessor.pkl')
        net = load_object('artifacts\model.pkl')
        X_test_preprocessed = preprocessor.get_data_transformation_object(data)
        net.eval()
        with torch.no_grad():
            outputs = net(X_test_preprocessed)
            _, predicted = torch.max(outputs, 1)
            return predicted
    except Exception as e:
        raise CustomException(e,sys)
    
# if __name__ == "__main__":
#     predict_test(test_data_path=r"B:\Projects\NLP_irrelevent_question\artifacts\test.csv",batch_size=64)