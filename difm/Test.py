import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":
    print("Loading...")
    target = ['label']
    fixlen_feature_columns = np.load("fixlen_feature_columns.npy",allow_pickle=True).tolist()
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    
    test = pd.read_csv("test.csv")
    print("test:")
    print(test.head())
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    
    
    
    test_model_input = {name: test[name] for name in feature_names}
    
    device = 'cuda:3'
    
    model = torch.load('jiaojiao1.h5')
    
    
    
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))