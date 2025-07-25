import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2

from tree_Agglomerative import MOCT
from read_data import read_data

def feature_select(x_train, y_train):

    sk = SelectKBest(chi2, k='all').fit(x_train, y_train)
    fea_index = np.argsort(- sk.scores_)

    return fea_index

for name in ["HT29", "A375", "A549"]:
    data = read_data(name)
    data = pd.DataFrame(data)
    x = data.drop(data.shape[1]-1, axis=1)
    y = data[data.shape[1]-1]
    if type(y[0]) == str:
        classes = list(set(y))
        y = y.replace({classes[i]: i for i in range(len(classes))})
    x, y = np.array(x), np.array(y)
    
    print("**************************{}*************************".format(name))
    print("Class distrbution: {}".format(dict(Counter(y))))
    print("data shape: {}".format(x.shape))
    
    aucs, f1s = [], []
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    
    k = 1
    for train_id, test_id in skf.split(x, y):
        print("----------Kfold-{}----------".format(k))
        x_train, x_test = x[train_id], x[test_id]
        y_train, y_test = y[train_id], y[test_id]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_train = np.array(x_train, type(float))
        x_test = scaler.transform(x_test)
        x_test = np.array(x_test, type(float))
            
        index = feature_select(x_train, y_train)
        x_train = x_train[:, index[:50]]
        x_test = x_test[:, index[:50]]
        
        clf = MOCT(max_depth=10, min_samples_leaf=0.1, method="SMOTE", n_jobs=16)
        clf.fit(x_train, y_train)
        
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_pred_proba, average="macro", multi_class='ovr')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        aucs.append(auc)
        f1s.append(f1_macro)
        
        k += 1
    
    print("AUC mean={:.4f}, std={:.4f}".format(np.mean(auc), np.std(aucs)))
    print("f1_score mean={:.4f}, std={:.4f}\n".format(np.mean(f1s), np.std(f1s)))
    
    