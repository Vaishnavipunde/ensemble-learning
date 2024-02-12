# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:04:24 2024

@author: rajendra
"""



import pandas as pd
df=pd.read_csv("c:\2-dataset\puma_diabetes")
df.columns
df.isnull().sum()
df.value_count()


X=df.drop("outcome",axis="columns")
y=df.outcomes


from sklearn.preprocessing import StandScalar
scaler=StandScalar()
X_scaled=scaler.fit_transform(X)
X_scaled[:3]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,strarify=y,random_state=10)





from sklearn.model_selection import cross_val_score
from skelearn.tree import DecisionTreeClassifier
scores=cross_val_score(DecisionTreeClassifier(), X,y,cv=5)
scores
scores.mean()


from sklearn.ensemble import BaggingClassifier
bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)

bag_model.fit(X_train,y_train)
bag_model.oob_score_
bag_model.score(X_test,y_test)
bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
scores=cross_val_score(bag_model, X,y,cv=5)
scores
scores.mean()


from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(n_estimators=50), X,y,cv=5)
scores.mean()


