# GRUPO 03
# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import projections
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn.metrics import r2_score
import random

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

    print('G) EQM1:' + str(eqm1) + '\nEQM2: ' + str(eqm2) + '\nEQM3: ' + str(eqm3) + '\nEQM8: ' + str(eqm8) + '\nQual é o mais preciso? EQM8')

    print('\nH)')
    
    #Divisão 90-10
    msk = np.random.rand(len(data)) < 0.9
    treino = data[msk]
    teste = data[~msk]

    xDadosDeTeste = np.array(teste[0])
    yDadosDeTeste = np.array(teste[1])

    xDadosDeTreinamento = np.array(treino[0])
    yDadosDeTreinamento = np.array(treino[1])

    print('\nX dados de teste' + str(xDadosDeTeste))
    print('\nY dados de teste' + str(yDadosDeTeste))

    print('\nX dados de treinamento' + str(xDadosDeTreinamento))
    print('\nY dados de treinamento' + str(yDadosDeTreinamento))

    print('\nI)')

    print('N = 1')

    betaI1 = np.polyfit(xDadosDeTreinamento, yDadosDeTreinamento, 1)
    regI1 = regressaoPolinominal(xDadosDeTreinamento, betaI1, 1)

    plt.scatter(xDadosDeTreinamento, yDadosDeTreinamento)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(betaI1))
    plt.plot(xDadosDeTreinamento, regI1, 'r')
    plt.show()

    print('N = 2')

    betaI2 = np.polyfit(xDadosDeTreinamento, yDadosDeTreinamento, 2)
    regI2 = regressaoPolinominal(xDadosDeTreinamento, betaI2, 2)

    plt.scatter(xDadosDeTreinamento, yDadosDeTreinamento)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(betaI2))
    plt.plot(xDadosDeTreinamento, regI2, 'g')
    plt.show()

    print('N = 3')

    betaI3 = np.polyfit(xDadosDeTreinamento, yDadosDeTreinamento, 3)
    regI3 = regressaoPolinominal(xDadosDeTreinamento, betaI3, 3)

    plt.scatter(xDadosDeTreinamento, yDadosDeTreinamento)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(betaI3))
    plt.plot(xDadosDeTreinamento, regI3, 'black')
    plt.show()

    print('N = 8')

    betaI8 = np.polyfit(xDadosDeTreinamento, yDadosDeTreinamento, 8)
    regI8 = regressaoPolinominal(xDadosDeTreinamento, betaI8, 8)

    plt.scatter(xDadosDeTreinamento, yDadosDeTreinamento)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafico de dispersão - Betas: ' + str(betaI8))
    plt.plot(xDadosDeTreinamento, regI8, 'y')
    plt.show()

    eqmI1 = eqm(yDadosDeTeste, regI1)
    eqmI2 = eqm(yDadosDeTeste, regI2)
    eqmI3 = eqm(yDadosDeTeste, regI3)
    eqmI8 = eqm(yDadosDeTeste, regI8)

    print('J) EQM1:' + str(eqmI1) + '\nEQM2: ' + str(eqmI2) + '\nEQM3: ' + str(eqmI3) + '\nEQM8: ' + str(eqmI8) + '\nQual é o mais preciso? EQM2')

    r2N1 = r2_score(yDadosDeTreinamento, regI1)
    r2N2 = r2_score(yDadosDeTreinamento, regI2)
    r2N3 = r2_score(yDadosDeTreinamento, regI3)
    r2N8 = r2_score(yDadosDeTreinamento, regI8)

    regI1Teste = regressaoPolinominal(xDadosDeTeste, betaI1, 1)
    regI2Teste = regressaoPolinominal(xDadosDeTeste, betaI2, 2)
    regI3Teste = regressaoPolinominal(xDadosDeTeste, betaI3, 3)
    regI8Teste = regressaoPolinominal(xDadosDeTeste, betaI8, 8)

    r2N1Teste = r2_score(yDadosDeTeste, regI1Teste)
    r2N2Teste = r2_score(yDadosDeTeste, regI2Teste)
    r2N3Teste = r2_score(yDadosDeTeste, regI3Teste)
    r2N8Teste = r2_score(yDadosDeTeste, regI8Teste)
  
    print('\nK) R2 dos dados de treinameto \nR2 n=1: ' + str(r2N1) + '\nR2 n=2: ' + str(r2N2) + '\nR2 n=3: ' + str(r2N3) + '\nR2 n=8: ' + str(r2N8))
    print('\nR2 dos dados de teste \nR2 n=1: ' + str(r2N1Teste) + '\nR2 n=2: ' + str(r2N2Teste) + '\nR2 n=3: ' + str(r2N3Teste) + '\nR2 n=8: ' + str(r2N8Teste))
