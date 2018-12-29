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



train.drop("Name",axis=1,inplace=True)
train.drop("Cabin",axis=1,inplace=True)
train.drop("Ticket",axis=1,inplace=True)

test.drop("Name",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
test.drop("Ticket",axis=1,inplace=True)


train_data = train.values
x_train = train_data[:, 2:]
y_train  = train_data[:, 1]

test_data = test.values
x_test = test_data[:, 1:]

id = y_test.values[:,0]
y_test = y_test.values[:,1]

#決定木
from sklearn import tree
from sklearn.grid_search import GridSearchCV

tuned_parameters = {'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'max_leaf_nodes':  [2,4,6,8,10]
                   }
 
clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0,splitter='best'), tuned_parameters, scoring="accuracy",cv=5, n_jobs=1)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test).astype(int)
with open("titanic_submit.csv", "w", newline='') as f:
    file = csv.writer(f)
    file.writerow(["PassengerId", "Survived"])
    file.writerows(zip(id, y_pred))
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


