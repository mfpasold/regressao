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

import faseUm


def regressaoMultipla(matrizX, beta):
    return np.matmul(matrizX, beta)


def obterBeta(matrizX, vetorY):
    beta = 0
    matrizXTransposta = np.array(matrizX).T
    mult = np.matmul(matrizXTransposta, matrizX)
    inversa = np.linalg.inv(mult)
    mult2 = np.matmul(matrizXTransposta, vetorY)
    beta = np.matmul(inversa, mult2)
    return beta


def dataDescribe():
    data = pd.read_csv("data.csv", header=None)

    desc = data.describe()

    print("Questão b:")
    print("Qual a média de preço das casas? R: " + str(desc.loc['mean'][2]))

    minCasa = data.loc[data[0] == desc.loc['min'][0]]
    print("Quanto custa a menor casa? R: " + str(minCasa.loc[44][2]))

    maxCasa = data.loc[data[2] == desc.loc['max'][2]]
    print("Quantos quartos tem a casa mais cara?? R: " + str(maxCasa.loc[13][1]))

    array = np.ones((47,3), dtype=int)

    print("\nQuestão c:")
    for i in range(47) :
        array[i][1] = data[0][i]
        array[i][2] = data[1][i]

    print(array)
    print('\n')

    tamCasa = [x[1] for x in array]
    noQuartos = [x[2] for x in array]
    cor1 = faseUm.correlacao(tamCasa, data[2])
    print(cor1)
    cor2 = faseUm.correlacao(noQuartos, data[2])
    print(cor2)
    beta = obterBeta(array, data[2])
    reg = regressaoMultipla(array, beta)
    print(reg)

    x = tamCasa
    y = noQuartos
    z = data[2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ax.scatter3D(x, y, z, color="green")
    ax.plot3D(x, y, reg)
    plt.title('Coeficiente de correlação entre Tamanho da casa e preço: ' + str(cor1)
              + '\nCoeficiente de correlação entre número de quartos e preço: ' + str(cor2))

    plt.show()

    array2 = np.ones((1, 3), dtype=int)

    print("\nQuestão c:")
    for i in range(1):
        array2[i][1] = 1650
        array2[i][2] = 3

    print('Tamanho 1650 e 3 quartos tem o preço de :' + str(regressaoMultipla(array2, beta)))

    regression = linear_model.LinearRegression()
    regression.fit(array, data[2])

    print('Os resultados da nossa implementação bateu com a biblioteca scikit-learn')
    print('Beta Nosso: ' + str(beta))
    print('Beta SciKit Learn: ')
    print(regression.intercept_)
    print(regression.coef_)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataDescribe()

