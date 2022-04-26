# GRUPO 03
# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import numpy
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

#Dataset
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]


def correlacao(vetorX, vetorY):
    lenX = len(vetorX)
    lenY = len(vetorY)
    mediaX = sum(vetorX) / lenX
    mediaY = sum(vetorY) / lenY
    dividendo = 0
    somaDivisorX = 0
    somaDivisorY = 0

    for i in range(lenX):
        dividendo += (vetorX[i] - mediaX) * (vetorY[i] - mediaY)
        somaDivisorX += (float(vetorX[i]) - mediaX) ** 2
        somaDivisorY += (float(vetorY[i]) - mediaY) ** 2

    divisor = numpy.sqrt(somaDivisorX * somaDivisorY)

    return round(dividendo / divisor, 4)


def calcularBeta0(vetorX, vetorY, beta1):
    lenX = len(vetorX)
    lenY = len(vetorY)
    mediaX = sum(vetorX) / lenX
    mediaY = sum(vetorY) / lenY

    return mediaY - beta1 * mediaX

def calcularBeta1(vetorX, vetorY):
    lenX = len(vetorX)
    lenY = len(vetorY)
    mediaX = sum(vetorX) / lenX
    mediaY = sum(vetorY) / lenY
    dividendo = 0
    divisor = 0

    for i in range(lenX):
        dividendo += (vetorX[i] - mediaX) * (vetorY[i] - mediaY)
        divisor += (float(vetorX[i]) - mediaX) ** 2

    return dividendo / divisor

def regressao(vetorX, vetorY):
    beta1 = calcularBeta1(vetorX, vetorY)
    print(round(beta1, 4))
    beta0 = calcularBeta0(vetorX, vetorY, beta1)
    print(round(beta0, 4))
    return round(beta0, 4), round(beta1, 4), [beta0 + beta1 * vetorX[x] for x in range(len(vetorX))]



if __name__ == '__main__':
    cor = correlacao(x1, y1)
    reg = regressao(x1, y1)

    str1 = 'Dataset 1 - Coeficiente de correlação : ' + str(cor) + ' β0 : ' + str(reg[0]) + ' β1 : ' + str(reg[1])
    plt.scatter(x1, y1)
    plt.plot(x1,reg[2], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str1)
    plt.show()

    cor = correlacao(x2, y2)
    reg = regressao(x2, y2)

    str1 = 'Dataset 2 - Coeficiente de correlação : ' + str(cor) + ' β0 : ' + str(reg[0]) + ' β1 : ' + str(reg[1])
    plt.scatter(x2, y2)
    plt.plot(x2, reg[2], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str1)
    plt.show()

    cor = correlacao(x3, y3)
    reg = regressao(x3, y3)

    str1 = 'Dataset 3 - Coeficiente de correlação : ' + str(cor) + ' β0 : ' + str(reg[0]) + ' β1 : ' + str(reg[1])
    plt.scatter(x3, y3)
    plt.plot(x3, reg[2], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str1)
    plt.show()


#Questão 3 - o terceiro dataset não é apropriado para regressão linear