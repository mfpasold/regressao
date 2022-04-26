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
        return beta[n] + beta[n-1] * x + beta[n-2] * np.power(x,2)
    if n == 3:
        return beta[n] + beta[n-1] * x + beta[n-2] * np.power(x,2) + beta[n-3] * np.power(x,3)
    if n == 8:
        return beta[n] + beta[n - 1] * x + beta[n - 2] * np.power(x,2) + beta[n - 3] * np.power(x,3) + beta[n - 4] * np.power(x,4) + beta[n - 5] * np.power(x,5) + beta[n - 6] * np.power(x,6) + beta[n - 7] * np.power(x,7) + beta[n - 8] * np.power(x,8)

def eqm(y, reg):
    total = 0
    for i in range(len(y)):
        total += ((y[i] - reg[i]) ** 2)
    return total / len(y)

if __name__ == '__main__':
    data = pd.read_csv("data_preg.csv", header=None)

    x = np.array(data[0])
    y = np.array(data[1])

    print('B)')

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão')
    plt.show()

    print('C)')

    beta1 = np.polyfit(x, y, 1)
    reg1 = regressaoPolinominal(x, beta1, 1)
    eqm1 = eqm(y, reg1)

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(beta1) + '\n Eqm: ' + str(eqm1))
    plt.plot(x, reg1, 'r')
    plt.show()

    print('D)')

    beta2 = np.polyfit(x, y, 2)
    reg2 = regressaoPolinominal(x, beta2, 2)
    eqm2 = eqm(y, reg2)

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(beta2) + '\n Eqm: ' + str(eqm2))
    plt.plot(x, reg2, 'g')
    plt.show()

    print('E)')

    beta3 = np.polyfit(x, y, 3)
    reg3 = regressaoPolinominal(x, beta3, 3)
    eqm3 = eqm(y, reg3)

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(beta3) + '\n Eqm: ' + str(eqm3))
    plt.plot(x, reg3, 'black')
    plt.show()

    print('F)')

    beta8 = np.polyfit(x, y, 8)
    reg8 = regressaoPolinominal(x, beta8, 8)
    eqm8 = eqm(y, reg8)

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(beta8) + '\n Eqm: ' + str(eqm8))
    plt.plot(x, reg8, 'y')
    plt.show()

    print('G) EQM1:' + str(eqm1) + '\nEQM2: ' + str(eqm2) + '\nEQM3: ' + str(eqm3) + '\nEQM8: ' + str(eqm8))

