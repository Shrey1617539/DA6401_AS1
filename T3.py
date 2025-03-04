from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd

def activation_function_cal(a, activation_function):
    if activation_function == 'sigmoid':
        return (1/(1+np.exp(-1*a)))
    
    if activation_function == 'tanh':
        return (np.exp(a)-np.exp(-1*a))/(np.exp(a)+np.exp(-1*a))
    
    if activation_function == 'ReLU':
        return a*(a>0)
    
    if activation_function == 'linear':
        return a
    
    if activation_function == 'softmax':
        a = np.exp(a)
        a /= np.sum(a)
        return a

def gradient_activation_function(a, activation_function):
    if activation_function == 'sigmoid':
        z = (1/(1+np.exp(-1*a)))
        return z*(1-z)
    
    if activation_function == 'tanh':
        z = (np.exp(a)-np.exp(-1*a))/(np.exp(a)+np.exp(-1*a))
        return (1 - (z**2))
    
    if activation_function == 'ReLU':
        return 1*(a>0)


def feedforward(x, weights, bias, activation_function):
    a = []
    h = []
    x = x.flatten()
    x = x.reshape(-1,1)

    for layer in range(len(weights)-1):
        x = bias[layer] + np.dot(weights[layer], x)
        a.append(x)
        x = activation_function_cal(x, activation_function[layer])
        h.append(x)

    x = bias[-1] + np.dot(weights[-1], x)
    a.append(x)
    x = activation_function_cal(x, activation_function[-1])
    h.append(x)

    return x, a, h
    
def loss_calculations(y, y_pred):
    if type(y) == list:
        y = np.array(y)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    loss = -1*np.log(y_pred[np.arange(y_pred.shape[0]), y, 0])
    loss = np.average(loss)
    return loss


def backpropagation(X, Y, weights, bias, activation_function):
    d_W = []
    d_b = []

    y_pred, a, h = feedforward(X, weights = weights, bias= bias, activation_function= activation_function)
    e_y = np.zeros(y_pred.shape)
    e_y[Y] = 1
    grad_al = (y_pred - e_y).reshape(-1,1)
    grad_hl = 0

    for layer in reversed(range(1, len(weights))):
        d_W.append(np.dot(grad_al, h[layer -1].T))
        d_b.append(grad_al)

        grad_hl = np.dot(weights[layer].T, grad_al)
        grad_al = grad_hl * gradient_activation_function(a[layer - 1], activation_function[layer - 1])
    
    d_W.append(np.dot(grad_al, X.flatten().reshape(-1,1).T))
    d_b.append(grad_al)

    d_W = list(reversed(d_W))
    d_b = list(reversed(d_b))
    
    return d_W, d_b
    
def reset_d_weights(weights, bias):
    d_W = []
    d_b = []

    for i in range(len(weights)):
        d_W.append(np.zeros(weights[i].shape))
        d_b.append(np.zeros(bias[i].shape))
    
    return d_W, d_b

def initialize_weights(input_size, hidden_layers, hidden_layer_size, output_size, initialisation):
    weights = []
    bias = []

    if initialisation == 'random':
        weights.append(np.random.randn(int(hidden_layer_size), int(input_size)))
        bias.append(np.random.randn(int(hidden_layer_size), 1))

        for i in range(hidden_layers-1):
            weights.append(np.random.randn(int(hidden_layer_size), int(hidden_layer_size)))
            bias.append(np.random.randn(int(hidden_layer_size), 1))

        weights.append(np.random.randn(int(output_size), int(hidden_layer_size)))
        bias.append(np.random.randn(int(output_size), 1))

    return weights, bias


def gradient_descent(X_data, Y_data, weights, bias, epochs, activation_function , learning_rate = 0.01, beta = 0, batch_size = None, optimization_method = None):
    if batch_size == None:
        batch_size = X_data.shape[0]

    for epoch in range(epochs):
        d_W, d_b = reset_d_weights(weights, bias)
        u_W, u_b = reset_d_weights(weights, bias)
        
        for i in range(X_data.shape[0]):
            X = X_data[i]
            Y = Y_data[i]

            if optimization_method == 'nesterov':
                for j in range(len(weights)):
                    weights[j] -= beta*u_W[j]
                    bias[j] -= beta*u_b[j]

            d_W_part, d_b_part = backpropagation(X, Y, weights = weights, bias = bias, activation_function = activation_function)

            for j in range(len(weights)):
                d_W[j] += d_W_part[j]
                d_b[j] += d_b_part[j]

            if optimization_method == None or optimization_method == 'gd':

                if (i+1)%batch_size == 0:
                    d_W = [x/batch_size for x in d_W]  
                    d_b = [x/batch_size for x in d_b]

                    for j in range(len(weights)):
                        weights[j] -= learning_rate * d_W[j]
                        bias[j] -= learning_rate * d_b[j]

                    d_W, d_b = reset_d_weights(weights, bias)

            if optimization_method == 'sgd':

                for j in range(len(weights)):
                    weights[j] -= learning_rate * d_W[j]
                    bias[j] -= learning_rate * d_b[j]

                d_W, d_b = reset_d_weights(weights, bias)
            
            if optimization_method == 'momentum':

                if (i+1)%batch_size == 0:
                    d_W = [x/batch_size for x in d_W]
                    d_b = [x/batch_size for x in d_b]

                    for j in range(len(weights)):
                        u_W[j] = beta*u_W[j] + learning_rate*d_W[j]
                        u_b[j] = beta*u_b[j] + learning_rate*d_b[j]

                        weights[j] -= u_W[j]
                        bias[j] -= u_b[j]
                    d_W, d_b = reset_d_weights(weights, bias)

            if optimization_method == 'nesterov':
                
                for j in range(len(weights)):
                    weights[j] += beta*u_W[j]
                    bias[j] += beta*u_b[j]

                if (i+1)%batch_size == 0:
                    d_W = [x/batch_size for x in d_W]
                    d_b = [x/batch_size for x in d_b]

                    for j in range(len(weights)):
                        u_W[j] = beta*u_W[j] + learning_rate*d_W[j]
                        u_b[j] = beta*u_b[j] + learning_rate*d_b[j]

                        weights[j] -= u_W[j]
                        bias[j] -= u_b[j]
                    d_W, d_b = reset_d_weights(weights, bias)
    return weights, bias