import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":

    target = ['label']
    fixlen_feature_columns = np.load("/home/liyaox/Xuhengyuan/Code/deepctr/criteo/fixlen_feature_columns.npy",allow_pickle=True).tolist()
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
        
    
    print("Loading...")
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    
    train = pd.read_csv("/home/liyaox/Xuhengyuan/Code/deepctr/criteo/train.csv")
    print("train:")
    print(train.head())
    
        
    train_model_input = {name: train[name] for name in feature_names}
    
    #cuda selection
    device = 'cuda'#:3'
    
    #model = torch.load('DeepFM.h5')
    
    model = difm.DIFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,att_head_num=4, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.295,  l2_reg_dnn=0.295,dnn_dropout=0, seed=1455,
                   task='binary',
                   l2_reg_embedding=0.295, device=device)
                   
    #torch.save(model,"DIFM_raw.h5")
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input,train[target].values,batch_size=1024,epochs=10,verbose=1,validation_split=0.2)
    print("**********Saving model...**********")
    
    #saving
    torch.save(model,"Feed_on_Totally_Random.h5")
    
    
    test = pd.read_csv("/home/liyaox/Xuhengyuan/Code/deepctr/criteo/test.csv")
    print("test:")
    print(test.head())
    test_model_input = {name: test[name] for name in feature_names}
    pred_ans = model.predict(test_model_input, 256)
    
    
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))