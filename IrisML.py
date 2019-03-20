#This programs uses Iris dataset to create a simple iris classifier

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Iris.csv')
#print(data)
'''
#Visualization of given data

plt.subplot(231)
plt.scatter(data.sepal_length.iloc[:50], data.sepal_width[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.sepal_length.iloc[50:100], data.sepal_width[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.sepal_length.iloc[100:], data.sepal_width[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.title('Sepal length - Sepal width')
plt.legend()


plt.subplot(232)
plt.scatter(data.sepal_length.iloc[:50], data.petal_length[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.sepal_length.iloc[50:100], data.petal_length[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.sepal_length.iloc[100:], data.petal_length[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Sepal length')
plt.ylabel('Petal length')

plt.title('Sepal length - Petal length')
plt.legend()

plt.subplot(233)
plt.scatter(data.sepal_length.iloc[:50], data.petal_width[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.sepal_length.iloc[50:100], data.petal_width[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.sepal_length.iloc[100:], data.petal_width[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Sepal length')
plt.ylabel('Petal width')

plt.title('Sepal length - Petal width')
plt.legend()

plt.subplot(234)
plt.scatter(data.sepal_width.iloc[:50], data.petal_length[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.sepal_width.iloc[50:100], data.petal_length[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.sepal_width.iloc[100:], data.petal_length[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')

plt.title('Sepal width - Petal length')
plt.legend()

plt.subplot(235)
plt.scatter(data.sepal_width.iloc[:50], data.petal_width[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.sepal_width.iloc[50:100], data.petal_width[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.sepal_width.iloc[100:], data.petal_width[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Sepal width')
plt.ylabel('Petal width')

plt.title('Sepal width - Petal width')
plt.legend()


plt.subplot(236)
plt.scatter(data.petal_length.iloc[:50], data.petal_width[:50],c = 'r', marker = '*', label = data.name.iloc[0])
plt.scatter(data.petal_length.iloc[50:100], data.petal_width[50:100],c = 'b', marker = '*',label = data.name.iloc[50])
plt.scatter(data.petal_length.iloc[100:], data.petal_width[100:],c = 'y', marker = '*',label = data.name.iloc[-1])

plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.title('Petal length - Petal width')
plt.legend()


plt.show()
'''



def sigmoid(x):
	return 1/(1+np.exp(-x))


def name_to_num(x, name):
	result = []
	for element in x:
		if element == name:
			result.append(1)
		else:
			result.append(0)
	return result


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def cost(h, y):
	result = (1/y.size) * (np.dot((-y), np.log(h)) 
			- np.dot((1 - y),(np.log(1 - h))))
	return result

def fit(X, Y, iter = 100000, lr = 0.03):
	X =  add_intercept(X)
	theta = np.zeros((X.shape[1],1))
	h = sigmoid(np.dot(X, theta))	
	for i in range(iter):
		h = sigmoid(np.dot(X, theta))
		gradient = (np.dot((h.T-Y),X) / Y.size).T
		theta -= lr*gradient
		h = sigmoid(np.dot(X, theta))
		if(i%10000 == 0):
			print(cost(h, Y))
	return theta

def predict_prob(theta, X):
	X = np.concatenate(([1], X), axis=0)
	result = sigmoid(np.dot(X, theta))
	return list(result).index(max(result))



theta = []
Newdata = data.drop(columns = ['name'])
for i in range(len(np.unique(data.name))):
	iris_Y = np.array(name_to_num(data.name, np.unique(data.name)[i]))
	iris_X = Newdata.values
	theta.append(fit(iris_X,iris_Y))

print(theta)

theta = np.array(theta)


test1 = np.array([5.4,3.3,1.3,0.3])
test2 = np.array([4.9,2.4,3.3,1.0])
test3 = np.array([6.1,3.0,4.9,1.8])
print(np.unique(data.name)[predict_prob(theta, test1)])
print(np.unique(data.name)[predict_prob(theta, test2)])
print(np.unique(data.name)[predict_prob(theta, test3)])
