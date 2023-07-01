import pandas as pd
import numpy as np
import os
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse

def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    #Reading the data
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(y_true, y_pred, pred_prob):
    """mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2"""

    accuracy = accuracy_score(y_true,y_pred)
    rocaucscore = roc_auc_score(y_true, pred_prob, multi_class='ovr')

    return accuracy, rocaucscore

def main(n_estimators, max_depth):
    df = get_data()
    train,test = train_test_split(df)

    X_train = train.drop(["quality"],axis=1)
    X_test = test.drop(["quality"],axis=1)
    
    y_train = train["quality"]
    y_test = test["quality"]

    # model training
    
    """lr = ElasticNet()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)"""
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)

        # model evaluation
        # mae, mse, rmse, r2 = evaluate(y_test, y_pred)
        accuracy, rocaucscore = evaluate(y_test,y_pred, pred_prob)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc_score", rocaucscore)

        #mlflow model logging
        mlflow.sklearn.log_model(rf,"randomforestmodel")

        # print(f"mae: {mae}, mse: {mse}, rmse: {rmse}, r2_score: {r2}")
        print(f"accuracy: {accuracy}")
        print(f"roc_auc_score: {rocaucscore}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n", default=50,type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parse_args = args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators, max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
