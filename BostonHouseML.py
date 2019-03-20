#This project creates model to predict house price 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('BostonHouse_train.csv')
data_test = pd.read_csv('BostonHouse_test.csv')
#print(data)
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
features_test = data_test.drop('MEDV', axis = 1)
prices_test = data_test['MEDV']
X = features.values
y = prices.values
y = np.reshape(y, (len(y), 1))
y = y/100000
X_test = features_test.values
y_test = prices_test.values
y_test = np.reshape(y_test, (len(y_test), 1))
y_test = y_test/100000
#print(features)

'''
#Shuffle dataframe and split into test and training sets(2:8)
Shuffled = data.sample(frac=1)
separator = 8*int(len(data)/10)

Shuffled[:separator].to_csv('BostonHouse_train.csv', index = False, encoding = 'utf-8')
Shuffled[separator:].to_csv('BostonHouse_test.csv', index = False, encoding = 'utf-8')
'''


#Create plots
'''
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
#print(data)

plt.figure(figsize=(20, 5))

for i,col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    x = data[col]
    y = prices
    plt.plot(x, y, '.')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')

plt.show()
'''


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def cost(X,Y, theta):
    m = len(Y)
    result = np.sum((np.dot(X,theta) - Y) ** 2)/(2 * m)
    return result


def cost_function(X, Y, theta):
    X = normalize(X)
    squared = np.power(X[:,0], 2)
    X = np.column_stack((X, squared))
    X = add_intercept(X)
    m = len(Y)
    J = np.sum((np.dot(X,theta) - Y) ** 2)/(2 * m)

    return J


def normalize(X):
    X = (X - np.mean(X, axis = 0))/(np.amax(X, axis = 0) - np.amin(X, axis = 0))
    return X


def hypothesis(X, theta):  
    return np.dot(X,theta)

def train(X, y, iter = 100000, lr = 1):
    X = normalize(X)
    squared = np.power(X[:,0], 2)
    X = np.column_stack((X, squared))
    X = add_intercept(X)  
    theta = np.ones((X.shape[1],1)) 

    for i in range(iter):
        h = hypothesis(X, theta)
        gradient = (lr/y.size)*(np.dot((h-y).T,X)).T
        theta -= gradient
        h = hypothesis(X, theta)
        if(i%60000 == 0):
            print(cost(X, y, theta))
    return theta


def normal_equation(X,y):
    inversed = np.linalg.inv(np.dot(X.T,X))
    theta = np.dot(inversed,X.T) 
    theta = np.dot(theta, y)
    return theta

theta_manual = train(X,y)
theta = normal_equation(X,y)
print(theta_manual)
print(cost_function(X_test, y_test, theta_manual))
print(cost(X, y, theta))
print(cost(X_test,y_test,theta))