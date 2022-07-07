import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint,reciprocal, uniform
from math import sqrt
from sklearn.svm import SVR
import matplotlib as mpl
import argparse
import joblib

parser = argparse.ArgumentParser(description='parameters of the model')

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#计算y_true 和y_predict 之间的准确率
def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)

# 计算均方误差MSE
def MSE(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

# 计算均方根误差RMSE
def RMSE(y_true, y_predict):
    return sqrt(MSE(y_true, y_predict))

# 计算平均绝对误差MAE
def MAE(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

# R Square
def r2_score(y_true, y_predict):
    return 1 - MSE(y_true, y_predict) / np.var(y_true)

def load_data():
    print('---------------loading data----------------')
    features = pd.read_csv('./data/MD_train_20.csv',header = 0)
    features = features.iloc[:, 1:]
    labels = pd.read_excel('./data/ERα_activity.xlsx',header = 0)
    labels = labels.iloc[:, 2]
    trainn_set, test_set, train_label, test_label = train_test_split(features, labels, test_size=.2,random_state=49)
    return  trainn_set, test_set, train_label, test_label


def load_model(name):
    print('---------------loading model----------------')
    if name == 'SVR':
        svm_clf = SVR(kernel='rbf')
        param_distributions = {"gamma": reciprocal(0.001, 0.01), "C": uniform(1, 10)}
        model = RandomizedSearchCV(svm_clf, param_distributions, n_iter=100,verbose=2,\
        cv=5,scoring='r2', random_state=42,n_jobs=1)

    if name == 'RF':
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=42)
        model = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,\
        n_iter=100, cv=5,scoring='r2', random_state=42)

    if name == 'GBRT':
        param_distribs = {
        'max_depth': randint(low=2, high=20),
        'n_estimators': randint(low=20, high=200),
        }
        gbrt = GradientBoostingRegressor(random_state=42,learning_rate=0.1)
        model = RandomizedSearchCV(gbrt, param_distributions=param_distribs,
        n_iter=100, cv=5, scoring='r2', random_state=42)

    return model


def train(name):
    X_train, X_val, y_train, y_val =load_data()
    model = load_model(name)
    model.fit(X_train, y_train)
    # 搜寻到的最佳模型
    print(model.best_estimator_)
    joblib.dump(model.best_estimator_,'./model/forest_reg_y.pkl') #将clf 存入.pkl 的文件中

    # 模型性能估计
    y_pred1 = model.best_estimator_.predict(X_train)
    y_pred2 = model.best_estimator_.predict(X_val)
    print("训练RMSE:",RMSE(y_train, y_pred1))
    print("训练MAE:",MAE(y_train, y_pred1))
    print("测试RMSE:",RMSE(y_val, y_pred2))
    print("测试MAE:",MAE(y_val, y_pred2))
    print('拟合度：',r2_score(y_val,y_pred2))

    #拟合效果图
    aa_true=pd.DataFrame(y_val[0:80]).values
    aa_val=y_pred2[0:80]
    plt.figure(figsize=(10,5.5))
    plt.plot(aa_true, "g.-",label="ture")
    plt.plot(aa_val, "r.-",label="val")
    plt.xlabel("indexes of data points")
    plt.ylabel("pIC50")
    plt.legend(['test_true','test_pre'])
    plt.savefig('./pic/prediction_vs_ground_truth_vaule_{}.png'.format(name), dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
    
    return model


def infer(name,model):
    X_test= pd.read_csv('./data/MD_test_20.csv' , header = 0)
    X_test=X_test.iloc[:,1:]
    print(X_test.shape)

    #预测50 组数据并保存
    y_test_pre_pIC50= model.predict(X_test)

    val=y_test_pre_pIC50
    plt.figure(figsize=(10,5.5))
    plt.plot(val, "r.-",label="test_pre")
    plt.xlabel("indexes of data points")
    plt.ylabel("pIC50")
    plt.legend(['test_pre'])
    plt.savefig('./pic/test_prediction_{}.png'.format(name), dpi=500, bbox_inches='tight')

    y_test_pre_pIC50=pd.DataFrame(y_test_pre_pIC50)
    y_test_pre_pIC50.to_csv('./data/{}_test_pre_pIC50.csv'.format(name), index=False)

    



parser.add_argument('--model', type=str, default='SVR', help='model type')
args = parser.parse_args()
model=train(args.model)
infer(args.model, model)
