#bibliotecas numéricas
from random import *
from math import *

#bibliotecas para plotagem
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#biblioteca para redes neurais
from sklearn.neural_network import MLPRegressor

#funcao para aproximação
def sombreiro(x, y, option):
	if(option == 'np_array'):
		return np.sin(np.sqrt(x**2+y**2))/np.sqrt(x**2+y**2)
	else:
		z = []
		for i in range(len(x)):
			z.append(sin(sqrt(x[i]**2+y[i]**2))/sqrt(x[i]**2+y[i]**2))
		return z


#parametros essenciais
training_goal = 0.0001
learning_rate = 0.01
number_of_neurons = 25
number_of_points = 250

#definicao das entradas para treinamento da rede
x = []
y = []
for i in range(0, number_of_points): 
	x.append(uniform(-10,10))
	y.append(uniform(-10,10))
x.sort()
y.sort()
X, Y = np.meshgrid(x, y)
Z = sombreiro(X, Y, 'np_array')

#saida para treinamento da rede
z = sombreiro(x, y, '')

#plot da superfície inicial
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

#definição da rede com BP
neural_network = MLPRegressor(
	hidden_layer_sizes = (number_of_neurons, number_of_neurons , number_of_neurons),
	solver = 'sgd',
	activation = 'logistic',
	learning_rate = 'constant',
	learning_rate_init = learning_rate,
	tol = training_goal
)

#formatando entrada para o scikit
train_input = []
aux = []
for i in range(0, number_of_points):
	aux.append(x[i])
	aux.append(y[i])
	train_input.append(aux)
	aux = []

#treinamento da rede
neural_network.fit(train_input, z)

#formatando dados para predição
predict_input = []
x_predict = []
y_predict = []

for i in range(0, number_of_points):
	x_predict.append(uniform(-10,10))
	y_predict.append(uniform(-10,10))

for i in range(0, number_of_points):
	aux.append(x_predict[i])
	aux.append(y_predict[i])
	predict_input.append(aux)
	aux = []

#predição do modelo
predicted_output = neural_network.predict(predict_input)

print(z, predicted_output)

#plot dos pontos preditos
ax.scatter(x_predict, y_predict, predicted_output, c='r', marker='o')
plt.show()
