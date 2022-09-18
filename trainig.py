import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import sys

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

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

def precission_recall(y_predict, y_true):
    TP = TN = FP = FN = 1

    for i in range(len(y_predict)):
        if (y_predict[i] == y_true[i] and y_predict[i] == 1):
            TP += 1
        elif (y_predict[i] == y_true[i] and y_predict[i] == 0):
            TN += 1
        elif (y_predict[i] != y_true[i] and y_predict[i] == 1):
            FP += 1
        elif (y_predict[i] != y_true[i] and y_predict[i] == 0):
            FN += 1

    precission = TP/(TP + FP)
    recall = TP/(TP + FN)
    return precission,recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Датасет для обучения модели', )
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)

    input_size = len(feauture_list)
    hidden_size = int(input_size/2)
    output_size = 2

    results = pd.DataFrame(columns=["mse", "accuracy", "f_score"])

    np.random.seed(20)

    # initializing weight for the first hidden layer
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

    # initializing weight for the second hidden layer
    W2 = np.random.normal(scale=0.5, size=(hidden_size, hidden_size)) 

    # initializing weight for the output layer
    W3 = np.random.normal(scale=0.5, size=(hidden_size, output_size)) 

    learning_rate = 0.1

    # input data shape
    line, col = X_train.shape

    # uteration_number
    n_iter = 1001

    for iter in range(n_iter):
        # feedforward
        A1 = feedforward(X_train, W1)   # hidden layer 1
        A2 = feedforward(A1, W2)        # hidden layer 2
        A3 = feedforward(A2, W3)        # output_layer

        # Calculating metrics
        mse = mean_squared_error(A3, y_train)
        acc = accuracy(A3, y_train)
        precission, recall = precission_recall(A3.argmax(axis=1), y_train.argmax(axis=1))
        f_score = 2 * ((precission * recall)/(precission / recall))
        print("epoch {0}/{1} - loss: {2} - acc: {3} - f_score: {4}".format(iter, n_iter - 1, mse, acc, f_score))
        if (mse < 0.015):
            break
        results=results.append({"mse":mse, "accuracy":acc, "f_score":f_score},ignore_index=True)

        # backprop
        E1 = A3 - y_train               # output_error
        dW3 = E1 * A3 * (1 - A3)

        E2 = np.dot(dW3, W3.T)          # hidden layer 2 error
        dW2 = E2 * A2 * (1 - A2)

        E3 = np.dot(dW2, W2.T)          # hidden layer 1 error
        dW1 = E3 * A1 * (1 - A1)

        # weight updates
        W3_update = np.dot(A2.T, dW3) / col         # output weights
        W2_update = np.dot(A1.T, dW2) / col         # hideen weights
        W1_update = np.dot(X_train.T, dW1) / col    # hideen weights

        W3 = W3 - learning_rate * W3_update
        W2 = W2 - learning_rate * W2_update
        W1 = W1 - learning_rate * W1_update


    results.accuracy.plot()
    results.mse.plot()
    results.f_score.plot()
    plt.show()

    results.to_csv("log_train.csv")

    L3 = softmax(feedforward(feedforward(feedforward(X_test, W1), W2), W3))
    print("Точнось на тестовых данных: {0}%\n".format(accuracy(L3, y_test) * 100))
    np.save("weights.npy", np.array([W1, W2, W3]))

if __name__ == '__main__':
    sys.exit(main())