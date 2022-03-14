import numpy as np
from config import cfg
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score


def pairwise_prediction(X_train,X_test,train_label,test_label):
    if 'svm' in cfg.classifier:
        ###svm###
        model = LinearSVC(loss='hinge', tol=0.00001, C=1.0).fit(X_train,train_label)
        test_pred = model.predict(X_test)
        svm_auc = roc_auc_score(test_label,test_pred)
    else:
        svm_auc = 0
    if 'mlp' in cfg.classifier:
        ###mlp###
        model = MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,early_stopping=True).fit(X_train,train_label)
        test_pred = model.predict(X_test)
        mlp_auc = roc_auc_score(test_label,test_pred)
    else:
        mlp_auc = 0
    if 'lg' in cfg.classifier:
        ###lg###
        model = LogisticRegression(solver='liblinear',n_jobs=16).fit(X_train,train_label)
        test_pred = model.predict(X_test)
        lg_auc = roc_auc_score(test_label,test_pred)
    else:
        lg_auc = 0
    return svm_auc,mlp_auc,lg_auc



