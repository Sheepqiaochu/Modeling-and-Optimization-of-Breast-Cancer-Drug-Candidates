import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from scipy.stats import reciprocal, uniform,randint
import matplotlib as mpl
import argparse
import joblib

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def load_data(y_number):
    print('---------------loading data----------------')
    features = pd.read_csv('./data/MD_train_dropzero.csv',header = 0)
    features = features.iloc[:, 1:]
    labels = pd.read_excel('./data/ADMET.xlsx',header = 0)
    label =labels.iloc[:,int(y_number)]
    trainn_set, test_set, train_label, test_label = train_test_split(features, label, test_size=.2,random_state=49)
    return  trainn_set, test_set, train_label, test_label


def train(y_number):
    y_list=['Caco-2','CYP3A4','hERG','HOB','MN']
    X1_train, X1_val, y1_train, y1_val = load_data(y_number)

    #SGD
    print('---------------SGD----------------')
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42,n_jobs=200)
    sgd_clf.fit(X1_train, y1_train)
    y1_scores_sgd=cross_val_predict(sgd_clf,X1_train,y1_train, cv=3,method="decision_function")
    fpr_sgd,tpr_sgd,thresholds_sgd=roc_curve(y1_train,y1_scores_sgd)
    accuracy_SGD_train=sum(cross_val_score(sgd_clf,X1_train,y1_train,cv=10,scoring="accuracy"))/10
    accuracy_SGD_val=sum(cross_val_score(sgd_clf,X1_val,y1_val,cv=10,scoring="accuracy"))/10

    #SVM
    print('---------------SVM----------------')
    svm_clf = SVC(decision_function_shape="ovr", gamma="auto")
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2,cv=3,n_jobs=1)
    rnd_search_cv.fit(X1_train, y1_train)
    # 搜索到的最佳模型
    print('SVM最佳模型:',rnd_search_cv.best_estimator_)
    print('最佳参数:',rnd_search_cv.best_score_)
    
    y1_scores_svm = cross_val_predict(rnd_search_cv.best_estimator_, X1_train, y1_train,cv=3,method="decision_function")
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(y1_train, y1_scores_svm)
    accuracy_SVM_train=sum(cross_val_score(rnd_search_cv.best_estimator_, X1_train, y1_train,cv=10, scoring="accuracy"))/10
    accuracy_SVM_val=sum(cross_val_score(rnd_search_cv.best_estimator_, X1_val, y1_val,cv=10, scoring="accuracy"))/10
    
    #RF
    print('------------------RF--------------------')
    param_distributions = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
    }
    forest_clf = RandomForestClassifier(random_state=42)
    rnd_search_cv_forest = RandomizedSearchCV(forest_clf, param_distributions, n_iter=100,verbose=2, cv=3,n_jobs=16)
    rnd_search_cv_forest.fit(X1_train, y1_train)

    forest_clf_y=rnd_search_cv_forest.best_estimator_
    joblib.dump(forest_clf_y,'./model/forest_clf_{}.pkl'.format(y_list[int(y_number)-1])) #保存模型
    # 搜索到的最佳模型
    print('RF最佳模型:',rnd_search_cv_forest.best_estimator_)
    print('最佳参数:',rnd_search_cv_forest.best_score_)
    
    accuracy_RF_train=sum(cross_val_score(rnd_search_cv_forest.best_estimator_,X1_train,\
    y1_train, cv=10, scoring="accuracy"))/10
    accuracy_RF_val=sum(cross_val_score(rnd_search_cv_forest.best_estimator_, X1_val, y1_val,\
    cv=10, scoring="accuracy"))/10
    cross_val_score(rnd_search_cv_forest.best_estimator_,X1_train,y1_train,cv=10,scoring="accuracy")
    
    y_probas_forest = cross_val_predict(rnd_search_cv_forest.best_estimator_, X1_train, y1_train,\
    cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y1_train,y_scores_forest)

    # ROC曲线、AUC值和准确率
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, color='r', linewidth=2, label='SVM')
    plt.plot(fpr_sgd, tpr_sgd, color='g', linewidth=2, label='SGD')
    plt.plot(fpr_forest, tpr_forest, color='b', linewidth=2, label='RF')
    plt.legend(loc='lower right')
    plt.legend(['SVM','SGD','RF'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
    plt.savefig('./pic/ROC_curve_{}.png'.format(y_number), dpi=500, bbox_inches='tight')
    plt.show()

    print('SGD的AUC值：',roc_auc_score(y1_train, y1_scores_sgd))
    print('SVM的AUC值：',roc_auc_score(y1_train, y1_scores_svm))
    print('RF的AUC值： ',roc_auc_score(y1_train, y_scores_forest))
    print('SGD 交叉验证训练集准确率： ',accuracy_SGD_train)
    print('SGD 交叉验证测试集准确率： ',accuracy_SGD_val)
    print('SVM 交叉验证训练集准确率： ',accuracy_SVM_train)
    print('SVM 交叉验证测试集准确率： ',accuracy_SVM_val)
    print('RF 交叉验证训练集准确率： ',accuracy_RF_train)
    print('RF 交叉验证测试集准确率： ',accuracy_RF_val)

    return forest_clf_y

def infer(model,number):
    X_test= pd.read_csv('./data/MD_test_dropzero.csv' , header = 0)
    X_test=X_test.iloc[:,1:]
    print(X_test.shape)

    # 预测test
    y_test_pred=model.predict(X_test)
    y_test_pred=pd.DataFrame(y_test_pred)
    y_test_pred.to_csv('./data/y{}_test_pred.csv'.format(number), index=False)

    return


parser = argparse.ArgumentParser(description='parameters of the model')
parser.add_argument('--number', type=str, default='1', help='y number(1-5)')
args = parser.parse_args()
model=train(args.number)
infer(model,args.number)
