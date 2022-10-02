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
    new_colume=['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']
    data = pd.read_csv('./train.txt',sep='\t',names=new_colume)
    print(data.head())
    
    print("Getting Features...")
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    print("Filling NaNs...")
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    print("Label Encoding...")
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    print("Counting Unique Features...")
    # 2.count #unique features for each sparse field,and record dense feature field name
    
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    m=np.array(fixlen_feature_columns)
    np.save('fixlen_feature_columns.npy',m)
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    """
    label_1_index = []   
    for i in range(1,len(train)):
        if train.iloc[i, 0] == 1:
            label_1_index.append(i)
    print(label_1_index)
    print(len(label_1_index))

    i = 0
    while i * 3 < len(train) and i < len(label_1_index):
        index = int(i * 3)      
        a, b = train.iloc[index, :].copy(), train.iloc[label_1_index[i], :].copy()
        train.iloc[index, :], train.iloc[label_1_index[i], :] = b, a
    
        i += 1
    """
    train, test = train_test_split(data, test_size=0.2)
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    print(test.head())
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cuda:0'
    """
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    """
    model = difm.DIFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)
    #torch.save(model,"DIFM_raw.h5")
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input,train[target].values,batch_size=256,epochs=10,verbose=1,validation_split=0.2)
    print("**********Saving model...**********")
    torch.save(model,"DIFM.h5")
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))