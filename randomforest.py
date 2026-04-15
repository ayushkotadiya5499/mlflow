import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

x,y=make_classification(n_samples=100,n_classes=2,n_redundant=2,n_informative=2,random_state=5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

n=10
m=10

with mlflow.start_run():

    x=RandomForestClassifier(n_estimators=n,max_depth=m)
    x.fit(x_train,y_train)
    y_pred=x.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',acc)
    print(acc)