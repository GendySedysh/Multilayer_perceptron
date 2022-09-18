from math import log
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import argparse
import sys

def standartize(x, mean, std):
    return (x - mean) / std

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*(y_pred).size) #(2*len(y_pred))

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

def softmax(y_values):
    exps = []
    for value in y_values:
        exps.append(np.exp(value))
    sum_of_exps = sum(exps)
    softmax = []
    for i in exps:
        softmax.append(i/sum_of_exps)
    return np.array(softmax)

def feedforward(input, W):
    Z = np.dot(input, W)
    A = sigmoid(Z)
    return A

def cross_entropy(y_predict, y_true):
    y = []
    target = []
    for i in range(len(y_predict)):
        if (y_predict[i][0] > y_predict[i][1]):
            y.append(y_predict[i][0])
        else:
            y.append(y_predict[i][1])
    
    for i in range(len(y_true)):
        if (y_true[i][0] > y_true[i][1]):
            target.append(1)
        else:
            target.append(0)

    loss_sum = 0
    for i in range(len(y)):
        loss_sum += target[i] * log(y[i], 10) + (1 - target[i]) * log(1 - y[i], 10)
    return -loss_sum / len(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Датасет для предсказания модели')
    parser.add_argument('weights', help='Файл с весами модели')
    args = parser.parse_args()

    # Load dataset
    columns = ["num", "label"]
    feauture_list = []
    # feauture_list = ['23', '3', '13', '22', '2', '20', '0', '12', '21', '1', '26', '10', '6', '25']
    for i in range(30):
        columns.append(str(i))
        feauture_list.append(str(i))
    data = pd.read_csv(args.dataset, names=columns)

    mean_std = {}
    for feat in feauture_list:
        mean_std[feat] = ({"mean":data[feat].mean(), "std":data[feat].std()})

    # Get features and target
    X=data[feauture_list]
    y=data.iloc[:,1]

    line, col = X.shape

    for i in range(line):
        for feat in feauture_list:
            tmp = X.at[i, feat]
            X.at[i, feat] = standartize(tmp, mean_std[feat]["mean"], mean_std[feat]["std"])

    # Get dummy variable 
    X = X.values
    y = pd.get_dummies(y).values

    # Get weights
    data = np.load(args.weights, allow_pickle=True)
    W1 = data[0]
    W2 = data[1]
    W3 = data[2]

    # Predict
    L3 = feedforward(feedforward(feedforward(X, W1), W2), W3)
    print("Точнось предсказания данных: {0}%".format(accuracy(L3, y) * 100))
    print("Показатель кросс-энтропии: {0}%\n".format(cross_entropy(L3, y)))
    answer = L3.argmax(axis=1)
    predict = []
    for i in answer:
        if (i == 1):
            predict.append("M")
        else:
            predict.append("B")

    pd.DataFrame(predict).to_csv("predictions.csv", header=False)

if __name__ == '__main__':
    sys.exit(main())