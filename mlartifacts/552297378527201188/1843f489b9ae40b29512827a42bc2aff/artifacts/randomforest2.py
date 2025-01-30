import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import mlflow
import seaborn as sns
from matplotlib import pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")

x,y=make_classification(n_samples=200,n_classes=2,n_redundant=2,n_informative=2,random_state=5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

n=15
m=10

mlflow.set_experiment('randomforest2')

with mlflow.start_run():

    x=RandomForestClassifier(n_estimators=n,max_depth=m)
    x.fit(x_train,y_train)
    y_pred=x.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',acc)
    mlflow.log_param('n_estimator',n)
    mlflow.log_param('max_depth',m)

    # confusion matrix
    c=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(c,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('actual')
    plt.ylabel('predict')
    plt.title('confusion matrix')
    plt.savefig('m.png')

    mlflow.log_artifact('m.png')
    mlflow.log_artifact(__file__)

    print(acc)