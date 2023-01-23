""" 
La llegada de vehículos a un estacionamiento sigue un proceso de Poisson de tasa λ vehículos/hora.
En el trabajo práctico se incluye un archivo con números pseudoaleatorios que representan los
tiempos entre arribos de dichos vehículos.

● Realice una estimación de la tasa de arribos y proponga un test para evaluar si los números
pseudoaleatorios provistos siguen la distribución esperada.
● Utilizando el GCL implementado en el Ejercicio 1 simular la llegada de vehículos durante un mes.
Graficar la cantidad de arribos en función del tiempo.
● Generar 1000 realizaciones del proceso de Poisson para estimar las siguientes probabilidades:
    1. Probabilidad que el primer vehículo arribe antes de los 10 minutos.
    2. Probabilidad que el undécimo vehículo arribe después de los 60 minutos.
    3. Probabilidad que arriben al menos 750 vehículos antes de las 72 horas.
    
Comparar con los valores teóricos.

"""

import matplotlib.pyplot as plt
import numpy as np
import time
import math
from scipy.stats import chi2, expon
from scipy import stats, mean

#FUNCIONES Y CLASES NECESARIAS:
class Xorshift():
    # El maximo se debe utilizar para limitar el tamaño del int en python
    max = 2**64
    seed = time.time_ns() % max

    def __xorshift__(self) -> int:
        x = self.seed
        x = x ^ (x << 13) % self.max
        x = x ^ (x >> 7) % self.max
        x = x ^ (x << 17) % self.max
        self.seed = x
        return x

    def rand(self, number, betweenZeroAndOne: bool):
        x = []
        for i in range(number):
            if(betweenZeroAndOne):
                x.append(self.__xorshift__()/self.max)
            else:
                x.append(self.__xorshift__())
        return x

def chi2_test(observed, amount, Pi): 
    alfa = 0.05

    chi2_teorico = chi2.ppf(1-alfa, amount-1)

    #Calculo de Chi2
    #print(observed)
    N = np.sum(observed)

    D2 = 0
    count = 0
    for obs in observed:
        D2 += ((obs - (N*Pi[count]))**2) / (N*Pi[count])
        count += 1

    #Se compara el limite conta el estadistico obtenido: D2 > chi2_teorico
    print("D2: ", D2)
    print("chi2_teorico: ", chi2_teorico)

    if D2 > chi2_teorico:
        print("Se rechaza Ho")
    else:
        print("No hay evidencia suficiente para rechazar Ho")

def exp_generator(n, lambda_p, xorshift):
    """
    Genera n puntos según distribución exponencial
    de parámetro lambda_p.
    """
    numeros = []
    numeros_u = xorshift.rand(n, True)

    for u in numeros_u:
        x = - (1/lambda_p) * math.log(1-u)
        numeros.append(x)
        
    return numeros


#Para estimar la tasa de arribos se tiene en cuenta la cantidad total de vehículos que arribaron
#al estacionamiento y el tiempo total, como la suma del tiempo de cada arribo. El parámetro
#lambda será: (ecuación)

#Para el tiempo total

with open('tiempos_entre_arribos.txt') as file:
    times = []
    for line in file:
        times.append(float(line.rstrip('\n')))
        
total_arrives = len(times)
total_time = sum(times)
lambda_p = 1 / np.mean(times)

print(f'Cantidad de arribos al estacionamiento: {total_arrives}')
print(f'Tiempo del último arribo: {total_time:.2f} horas')
print(f'Parámetro lambda: {lambda_p:.0f} vehiculos/hora')


#Test elegido: Chi2. Se discretiza los tiempos de arribos registrados en 10 clases
#(o grupos), en función del máximo tiempo, max(times)

groups = []

max_time = max(times)

for i in range(11):
    groups.append((i / 10) * max_time)

observations = np.zeros(10)

for time in times:
    for j in range(10):
        if groups[j] < time <= groups[j + 1]:
            observations[j] += 1

#Se calculan las probabilidades de cada grupo

group_prob = []

for i in range(len(groups) - 1):
    prob_a = expon.cdf(groups[i], loc=0, scale=1/lambda_p)
    prob_b = expon.cdf(groups[i + 1], loc=0, scale=1/lambda_p)
    group_prob.append(prob_b - prob_a)

chi2_test(observations, 10, group_prob)

#Se simula la llegada de autos a partir del generador Xorshift. Para ello, se
#define la función exp_generator que genera n valores con distribución exponencial
#de parámetro lambda_p usando Xorshift como punto de partida.

xorshift = Xorshift()
exp_arrives = exp_generator(14400, lambda_p, xorshift)
poisson_arrives = [0]

for i in range(len(exp_arrives)):
    poisson_arrives.append(poisson_arrives[i] + exp_arrives[i])
    
plt.figure(figsize=(10, 10))    
plt.step(poisson_arrives, range(len(poisson_arrives)), where= 'post' , label='λ=10 veh/h')
plt.xlabel('Cantidad de arribos')
plt.ylabel('Tiempo [horas]')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10)) 
plt.step(poisson_arrives[:21], range(len(poisson_arrives[:21])), where= 'post' , label='λ=10 veh/h')
plt.xlabel('Cantidad de arribos')
plt.ylabel('Tiempo [horas]')
plt.legend()
plt.show()

#Cálculo de las probabilidades teóricas, asumiendo parámetro lambda calculado anteriormente

#A:
lambda_t = 1/6 * lambda_p
poisson = stats.poisson(lambda_t)
theo_a = 1 - poisson.pmf(0)

#B:
lambda_t = 1 * lambda_p
poisson = stats.poisson(lambda_t)
theo_b = 0
for i in range(11):
    theo_b += poisson.pmf(i)

#C
lambda_t = 72 * lambda_p
poisson = stats.poisson(lambda_t)
theo_c = 1
for i in range(750):
    theo_c -= poisson.pmf(i)
    
#Cálculo experimental: se generan 1000 experimentos, simulando el arribo de 750 vehículos
#con parámetro lambda calculado anteriormente. Para cada evento se define un contador particular:

#cont_a = cantidad de experimentos en los cuales llega un vehículo antes de los 10 minutos.
#cont_b = cantidad de experimentos en los cuales llegan 10 vehículos o menos antes de la hora.
#cont_c = cantidad de experimentos en los cuales llegan 750 vehículos o menos antes de las 72 horas.

EXPERIMENTS = 1000

cont_a = 0
cont_b = 0
cont_c = 0

for i in range(EXPERIMENTS):
    exp_arrives = exp_generator(750, lambda_p, xorshift)
    poisson_arrives = [0]

    for j in range(len(exp_arrives)):
        poisson_arrives.append(poisson_arrives[j] + exp_arrives[j])
    
    if (poisson_arrives[1] < (1/6)):
        cont_a += 1
    
    if poisson_arrives[11] > 1:
        cont_b += 1
        
    if poisson_arrives[750] <= 72:
        cont_c += 1
        
exp_a = cont_a / EXPERIMENTS
exp_b = cont_b / EXPERIMENTS
exp_c = cont_c / EXPERIMENTS

print()
print(f'Probabilidad teórica del evento A: {theo_a:0.4f}')
print(f'Probabilidad experimental del evento A: {exp_a:0.4f}')
print(f'Error relativo del cálculo: {(abs(theo_a - exp_a)*100/theo_a):0.2f}%')
print()
print(f'Probabilidad teórica del evento B: {theo_b:0.4f}')
print(f'Probabilidad experimental del evento B: {exp_b:0.4f}')
print(f'Error relativo del cálculo: {(abs(theo_b - exp_b)*100/theo_b):0.2f}%')
print()
print(f'Probabilidad teórica del evento C: {theo_c:0.4f}')
print(f'Probabilidad experimental del evento C: {exp_c:0.4f}')
print(f'Error relativo del cálculo: {(abs(theo_c - exp_c)*100/theo_c):0.2f}%')
print()

    
