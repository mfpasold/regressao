import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import projections
from mpl_toolkits import mplot3d
from sklearn import linear_model

def regressaoPolinominal(x, beta, n):
    if n == 1:
        return beta[n] + beta[n-1] * x
    if n == 2:
        return beta[n] + beta[n-1] * x + np.power((beta[n-2] * x),2)
    if n == 3:
        return beta[n] + beta[n-1] * x + (beta[n-2] * x) ** 2 + (beta[n-3] * x) ** 3
    if n == 8:
        return beta[n] + beta[n - 1] * x + (beta[n - 2] * x) ** 2 + (beta[n - 3] * x) ** 3 + (beta[n - 4] * x) ** 4 + (beta[n - 5] * x) ** 5 + (beta[n - 6] * x) ** 6 + (beta[n - 7] * x) ** 7 + (beta[n - 8] * x) ** 8

def eqm(y, reg):
    total = 0
    for i in range(len(y)):
        total += ((y[i] - reg[i]) ** 2)
    return total / len(y)

if __name__ == '__main__':
    data = pd.read_csv("data_preg.csv", header=None)

    print('B)')

    plt.scatter(data[0], data[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão')
    plt.show()

    print('C)')

    beta1 = np.polyfit(data[0], data[1], 1)
    print(beta1)
    reg1 = regressaoPolinominal(data[0], beta1, 1)
    mse1 = eqm(data[1], reg1)
    print(mse1)

    plt.scatter(data[0], data[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão')
    plt.plot(data[0], reg1, 'r')
    plt.show()

    print('D)')

    beta2 = np.polyfit(data[0], data[1], 2)
    print(beta2)
    reg2 = regressaoPolinominal(data[0], beta2, 2)
    mse2 = eqm(data[1], reg2)
    print(mse2)

    mse = np.square(np.subtract(data[1], reg2)).mean()
    print(mse)

    plt.scatter(data[0], data[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão')
    plt.plot(data[0], reg2, 'g')
    plt.show()

