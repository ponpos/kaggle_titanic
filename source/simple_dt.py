# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv as csv

#Sex male:0, female:1
#Embarked S:0, C:1, Q:2

train = pd.read_csv('train.csv').replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test = pd.read_csv('test.csv').replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
y_test = pd.read_csv('gender_submission.csv')


train["Age"].fillna(train.Age.mean(),inplace=True)
train["Embarked"].fillna(train.Embarked.mean(),inplace=True)

test["Age"].fillna(train.Age.mean(),inplace=True)
test["Embarked"].fillna(train.Embarked.mean(),inplace=True)
test["Fare"].fillna(train.Fare.mean(), inplace=True)

#家族の総人数を追加
train["FamilySize"]=train["SibSp"]+train["Parch"]+1
test["FamilySize"]=test["SibSp"]+test["Parch"]+1

#Nameから敬称を抜き取る
def name_classifier(name_df):    
    name_class_df = pd.DataFrame(columns=['miss','mrs','master','mr'])
    
    for name in name_df:        
        if 'Miss.' in name:
            df = pd.DataFrame([[1.0,0.0,0.0,0.0]],columns=['miss','mrs','master','mr'])
        elif 'Mrs.' in name:
            df = pd.DataFrame([[0.0,1.0,0.0,0.0]],columns=['miss','mrs','master','mr'])
        elif 'Master.' in name:
            df = pd.DataFrame([[0.0,0.0,1.0,0.0]],columns=['miss','mrs','master','mr'])
        elif 'Mr.' in name:
            df = pd.DataFrame([[0.0,0.0,0.0,1.0]],columns=['miss','mrs','master','mr'])
        else :
            df = pd.DataFrame([[0.0,0.0,0.0,0.0]],columns=['miss','mrs','master','mr'])
        name_class_df = name_class_df.append(df,ignore_index=True)        
    return name_class_df

testes = name_classifier('Mr.')

train_nameclass = name_classifier(train["Name"])
train=pd.concat([train,train_nameclass],axis=1)

test_nameclass = name_classifier(test["Name"])
test=pd.concat([test,test_nameclass],axis=1)

train.drop("Name",axis=1,inplace=True)
train.drop("Cabin",axis=1,inplace=True)
train.drop("Ticket",axis=1,inplace=True)

test.drop("Name",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
test.drop("Ticket",axis=1,inplace=True)

print("check1")

train_data = train.values
print("check2")
x_train = train_data[:, 2:]
print("check3")
y_train  = train_data[:, 1]

test_data = test.values
x_test = test_data[:, 1:]

id = y_test.values[:,0]
y_test = y_test.values[:,1]


#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier 
from sklearn.grid_search import GridSearchCV

parameters = {
    "n_estimators":[i for i in range(10,100,10)],
    "criterion":["gini","entropy"],
    "max_depth":[i for i in range(1,6,1)],
     'min_samples_split': [2, 4, 10,12,16],
    "random_state":[3],
}
#モデルを作成
clf = GridSearchCV(RandomForestClassifier(), parameters,cv=5,n_jobs=-1)
clf_fit=clf.fit(x_train,y_train)
#最も良い学習モデルで学習
predictor=clf_fit.best_estimator_

y_pred = predictor.predict(x_test).astype(int)
with open("titanic_submit.csv", "w", newline='') as f:
    file = csv.writer(f)
    file.writerow(["PassengerId", "Survived"])
    file.writerows(zip(id, y_pred))
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


