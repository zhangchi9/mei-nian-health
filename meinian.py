import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
pd.options.mode.chained_assignment = None

def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ss

def getnum(string): 
    num = []
    for s in jieba.lcut(string, cut_all=False):
        l = []
        for t in s.split():
            try:
                l.append(float(t))
            except ValueError:
                pass
        if len(l) > 0:
            num.append(l[0])
    
    return num
    

data1 = pd.read_csv('meinian_round1_data_part1_20180408.txt',delimiter="$")
data2 = pd.read_csv('meinian_round1_data_part2_20180408.txt',delimiter="$")
train = pd.read_csv('meinian_round1_train_20180408.csv', encoding = "ISO-8859-1")
test = pd.read_csv('meinian_round1_test_b_20180505.csv', encoding = "ISO-8859-1")


data = data1.append(data2, ignore_index=True)

data= pd.pivot_table(data, values='field_results', index='vid',columns='table_id', aggfunc=lambda x: ' '.join(str(v) for v in x))
data.reset_index(inplace=True)
data.set_index('vid',inplace=True)

(rowlen,collen) = data.shape
d = {}
for i in range(collen):
    d[i] = 1 - (data.iloc[:,i].isnull().sum()/57298)
x = list(d.values())
x = sorted(x, reverse=True)
plt.plot(x)
plt.show()

index_keep = [k for k in d if d[k]>0.8]

data_k = data.iloc[:,index_keep]

col_0420_rep = {'未见异常':0, '正常':0, '未闻及异常':0, '正常 正常':0, '有力':1, '心音弱':2, '心音遥远':2, '较低':2,
       '强弱不等':3, '心音强':1, '心音弱, 心音遥远':2, '心音强, 心音遥远':1, '右位心':0, '心音遥远, 心音弱':2,
       '..........................................................................................0':0,
       '主动脉第2心音强':1, '弱':2, '第二心音分裂':2, '低钝':2}
data_k.replace({"0420": col_0420_rep},inplace=True)

col_2302_rep = {'健康':0, '疾病':2, 'nan':0, '亚健康':1, ' 健康':0, '正常疲劳反应 健康':0,
       '健康？？？？？？？？？？？？？？？':0, '健康 健康':0, '6C17002059健康':0, '.健康':0,
       '6C17002185健康':0, '6C17002193健康':0, '6C17001723健康':0, '6C17002177健康':0,
       '.+健康':0, '3健康':0, '、健康':0, '6C17002207健康':0, 'Z117114765健康':0,
       'Z117078437健康':0,
       '                                                          健康':0,
       'I117110034健康':0, '                        健康':0, '261Quite34869健康':0,
       'P617026382健康':0, 'I117105807健康':0, 'Q21701909%健康':0, '671809健康':0,
       'Z117136428健康':0, '98健康':0, '1、289965健康':0, '5283698738881健康':0,
       'Z117053064健康':0, 'y疾病':2, 'P617026234健康':0, '7817076622健康':0, '肥健康':1,
       '                                                                健康':0}
data_k.replace({"2302": col_2302_rep},inplace=True)

col_3190_rep = {'-':0, '阴性':0, '+-':1, '+':2, '++':3, '+++':4, '-        0mmol/L':0, '2+':3,
       '0(-)':0, '阳性(+)':2, '++++':5, '2.8(+-)':1, '未做':0, '3+':4, '--':0, '阳性(1+)':2,
       '+1     5.5mmol/L':2, '≥55(+4)':5, '阳性3+':4}
data_k.replace({"3190": col_3190_rep},inplace=True)

col_3191_rep = {'-':0, '阴性':0, '+':2, '-        0umol/L':0, '0(-)':0, '+++':4, '未做':0,
       '阳性(+)':2, '8.6(+1)':2, '++':3, '+-':1, '2+':3, 'Normal':0}
data_k.replace({"3191": col_3191_rep},inplace=True)

col_3192_rep = {'-':0, '阴性':0, '+-':1, '+':2, '-        0mmol/L':0, '0(-)':0, '++':3, '未做':0,
       '+++':4, '0.5(+-)':1, '1.5(+1)':2, '4.0(+2)':3, '1+':2, '--':0, '阳性(+)':2,
       '+1     1.5mmol/L':2, '+-     0.5mmol/L':1}
data_k.replace({"3192": col_3192_rep},inplace=True)

col_3195_rep = {'-':0, '阴性':0, '+-':1, '+':2, '-           0g/L':0, '2+':3, '++':3, '0(-)':0, '阳性(+)':2, '+++':4, '未做':0, '3-5':4, '1+':2, '0.15(+-)':1, '+-       0.15g/L':1,   '0.3(+1)':2, '+1        0.3g/L':2, '1.0(+2)':3, '阳性(1+)':2, '+1':2}
data_k.replace({"3195": col_3195_rep},inplace=True)

col_3197_rep={'-':0, '阴性':0, '+':2, '未做':0, '+-':1, '--':0, '1+':2}
data_k.replace({"3197": col_3197_rep},inplace=True)

colname = list(data_k)

trans2num = [12,15,17,18,19,20,21,22,24,25,26,31]
for colind in trans2num:
    a = list(data_k.iloc[:,colind])
    b = []
    for ele in a:
        string = str(ele)
        if string is None:
            b.append(np.nan)
        elif string=='nan':
            b.append(np.nan)
        else:
            string = strQ2B(string)
            string = ''.join(string)
            l = getnum(string)
            if len(l) == 0:
                b.append(np.nan)
            else:
                meanl = sum(l) / float(len(l))
                b.append(meanl)
    data_k[colname[colind]+'new'] = b
    
data_k.loc[data_k['2403new'] == data_k['2403new'].max(),['2403new']] = 57.142
data_k['BMI'] = data_k['2403new']/data_k['2404new']**2*10000

xfeature = [colname[i]+'new' for i in trans2num] 
xfeature +=["0420","2302","3190","3191","3192","3195","3197",'BMI']

data_f = data_k.loc[:,xfeature]


data_f.replace('None', np.nan, inplace=True)
for head in xfeature:
    data_f[head].fillna(data_f[head].value_counts().index[0], inplace = True)

for colind in list(train):
    a = list(train.loc[:,colind])
    b = []
    for ele in a:
        string = str(ele)
        if string is None:
            b.append(np.nan)
        elif string=='nan':
            b.append(np.nan)
        else:
            string = strQ2B(string)
            string = ''.join(string)
            l = getnum(string)
            if len(l) == 0:
                b.append(np.nan)
            else:
                meanl = sum(l) / float(len(l))
                b.append(meanl)
    train[colind+'new'] = b
yfeature = ['Anew','Bnew','Cnew','Dnew','Enew']
train_f = train.loc[:,yfeature]

low = .001
high = .999
quant_df = train_f.quantile([low, high])
train_f = train_f.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & (x < quant_df.loc[high,x.name])], axis=0)

for head in yfeature:
    train_f.fillna(train_f[head].value_counts().index[0], inplace = True)

index = train.vid
Xtrain = data_f.loc[index,xfeature]
Ytrain = train_f.copy()

index2 = test.vid
Xtest = data_f.loc[index2,xfeature]

def cus(clf,x,y):
    ypred = clf.predict(x)
    return np.mean((np.log(np.abs((ypred+1)/(y+1)))**2))

n_folds = 5
def cv(model,X,y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= cross_val_score(model, X, y, scoring=cus, cv = kf)
    return(rmse)
    
GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15,min_samples_split=10, 
                                   loss='huber', random_state =5)

for head in yfeature:
    score = cv(GBoost, Xtrain, Ytrain[head])
    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
ypred = pd.DataFrame()
ypred['vid'] = test.vid
for head in yfeature:
    GBoost.fit(Xtrain, Ytrain[head])
    ypred[head] = GBoost.predict(Xtest)
    
ypred.to_csv('submit.csv',index=False,header=False)
    




