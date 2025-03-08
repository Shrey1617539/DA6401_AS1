from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import wandb

def activation_function_cal(a, activation_function):
    if activation_function == 'sigmoid':
        pos_mask = (a >= 0)
        neg_mask = (a < 0)

        result = np.zeros(a.shape)
        result[pos_mask] = 1 / (1 + np.exp(-a[pos_mask]))
        result[neg_mask] = np.exp(a[neg_mask]) / (1 + np.exp(a[neg_mask]))

        return result
    
    if activation_function == 'tanh':
        pos_mask = (a >= 0)
        neg_mask = (a < 0)

        result = np.zeros(a.shape)
        result[pos_mask] = (1 - np.exp(-2*a[pos_mask]))/ (1 + np.exp(-2*a[pos_mask]))
        result[neg_mask] = (np.exp(2*a[neg_mask])-1) / (np.exp(2*a[neg_mask])+1)

        return result
    
    if activation_function == 'ReLU':
        return a*(a>0)
    
    if activation_function == 'linear':
        return a
    
    if activation_function == 'softmax':
        a = np.exp(a)
        b = np.sum(a, axis = 1)
        b = b[:, np.newaxis, :]
        a = a/b
        return a

def gradient_activation_function(a, activation_function):
    if activation_function == 'sigmoid':
        z = activation_function_cal(a, activation_function=activation_function)
        return z*(1-z)
    
    if activation_function == 'tanh':
        z = activation_function_cal(a, activation_function=activation_function)
        return (1 - (z**2))
    
    if activation_function == 'ReLU':
        return 1*(a>0)


def feedforward(x, weights, bias, activation_function):
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],1)
    a = []
    h = []

    for layer in range(len(weights)-1):
        x = bias[layer] + ( weights[layer] @ x )
        a.append(x)
        x = activation_function_cal(x, activation_function[layer])
        h.append(x)

    x = bias[-1] + (weights[-1] @ x)
    a.append(x)
    x = activation_function_cal(x, activation_function[-1])
    h.append(x)

    return x, a, h
    
def loss_calculations(y, y_pred, loss_type = 'cross_entropy'):
    if type(y) == list:
        y = np.array(y)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if loss_type == 'cross_entropy':
        loss = -1*np.log(y_pred[np.arange(y_pred.shape[0]), y, 0])
        loss = np.average(loss)
    if loss_type == 'mean_squared_error' or loss_type == 'mse':
        y_array = np.zeros(y_pred.shape)
        y_array[np.arange(y_pred.shape[0]), y] = 1
        loss = np.sum((y_array - y_pred)**2, axis = 1)
        loss = np.mean(loss)
    return loss

def accuracy_calculations(y, y_pred):
    if type(y) == list:
        y = np.array(y)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    y_pred = np.argmax(y_pred, axis = 1)
    y_pred = y_pred.flatten()
    return np.sum(y_pred == y)/y.shape[0]

def backpropagation(X, Y, weights, bias, activation_function, loss_type = 'cross_entropy'):

    d_W = []
    d_b = []

    y_pred, a, h = feedforward(X, weights = weights, bias= bias, activation_function= activation_function)

    e_y = np.zeros(y_pred.shape)
    e_y[np.arange(e_y.shape[0]),Y] = 1

    if loss_type == 'cross_entropy':
        grad_al = (y_pred - e_y).reshape(e_y.shape[0], e_y.shape[1], 1)

    if loss_type == 'mean_squared_error' or loss_type == 'mse':
        sum_c = (y_pred - e_y)*y_pred
        sum_c = np.sum(sum_c, axis = 1)
        sum_c = sum_c[:, np.newaxis, :]
        grad_al = 2*y_pred*(y_pred - e_y - sum_c).reshape(e_y.shape[0], e_y.shape[1], 1)
    grad_hl = 0

    for layer in reversed(range(1, len(weights))):
        d_W.append(np.matmul(grad_al, h[layer - 1].transpose(0,2,1)))
        d_b.append(grad_al)

        grad_hl = np.matmul(weights[layer].T, grad_al)
        grad_al = grad_hl * gradient_activation_function(a[layer - 1], activation_function[layer - 1])
    
    d_W.append(np.matmul(grad_al, X.reshape(X.shape[0], X.shape[1]*X.shape[2],1).transpose(0,2,1)))
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
        weights.append(np.random.randn(int(hidden_layer_size[0]), int(input_size)))
        bias.append(np.random.randn(int(hidden_layer_size[0]), 1))

        for i in range(hidden_layers-1):
            weights.append(np.random.randn(int(hidden_layer_size[i+1]), int(hidden_layer_size[i])))
            bias.append(np.random.randn(int(hidden_layer_size[i+1]), 1))

        weights.append(np.random.randn(int(output_size), int(hidden_layer_size[-1])))
        bias.append(np.random.randn(int(output_size), 1))

    if initialisation == 'Xavier':
        weights.append(np.random.randn(int(hidden_layer_size[0]), int(input_size))*np.sqrt(2/(int(hidden_layer_size[0]) + int(input_size))))
        bias.append(np.random.randn(int(hidden_layer_size[0]), 1))

        for i in range(hidden_layers-1):
            weights.append(np.random.randn(int(hidden_layer_size[i+1]), int(hidden_layer_size[i]))*np.sqrt(2/(int(hidden_layer_size[i+1]) + int(hidden_layer_size[i]))))
            bias.append(np.random.randn(int(hidden_layer_size[i+1]), 1))

        weights.append(np.random.randn(int(output_size), int(hidden_layer_size[-1]))*np.sqrt(2/(int(output_size) + int(hidden_layer_size[-1]))))
        bias.append(np.random.randn(int(output_size), 1))

    return weights, bias


def gradient_descent(X_data, Y_data, weights, bias, epochs, activation_function , 
                     learning_rate = 0.01, beta = 0.5, beta1 = 0.5, beta2 = 0.5, weight_decay = 0,
                     momentum = 0.5, batch_size = None, optimization_method = None,
                     epsilon = 0.000001, loss_type = 'cross_entropy', X_val = None, Y_val = None, X_test = None, 
                     Y_test = None, logging_train = False, logging_val = False, logging_test = False):
    
    if batch_size == None:
        batch_size = X_data.shape[0]
    
    if optimization_method == 'sgd':
        batch_size = 1

    step_cnt = 0
    for epoch in range(epochs):

        indices = np.random.permutation(X_data.shape[0])
        X_data = X_data[indices]
        Y_data = Y_data[indices]

        d_W, d_b = reset_d_weights(weights, bias)

        for i in range(0, X_data.shape[0], batch_size):

            step_cnt += 1
            X = X_data[i:min(X_data.shape[0], i+batch_size)]
            Y = Y_data[i:min(X_data.shape[0], i+batch_size)]

            if optimization_method == 'nag':

                if epoch == 0 and i == 0:
                    u_W, u_b = reset_d_weights(weights, bias)
                
                for j in range(len(weights)):
                    weights[j] -= momentum*u_W[j]
                    bias[j] -= momentum*u_b[j]

            d_W_part, d_b_part = backpropagation(X, Y, weights = weights, bias = bias, activation_function = activation_function, loss_type=loss_type)

            for j in range(len(weights)):

                d_W[j] += np.mean(d_W_part[j], axis = 0)
                d_b[j] += np.mean(d_b_part[j], axis = 0)

            if optimization_method == None or optimization_method == 'gd':

                for j in range(len(weights)):
                    weights[j] -= learning_rate * (d_W[j] + weight_decay*weights[j])
                    bias[j] -= learning_rate * d_b[j]

                d_W, d_b = reset_d_weights(weights, bias)
            
            if optimization_method == 'momentum':

                if epoch == 0 and i == 0:
                    u_W, u_b = reset_d_weights(weights, bias)

                for j in range(len(weights)):
                    u_W[j] = momentum*u_W[j] + learning_rate*(d_W[j] + weight_decay*weights[j])
                    u_b[j] = momentum*u_b[j] + learning_rate*d_b[j]

                    weights[j] -= u_W[j]
                    bias[j] -= u_b[j]

                d_W, d_b = reset_d_weights(weights, bias)

            if optimization_method == 'nag':

                for j in range(len(weights)):
                    u_W[j] = momentum*u_W[j] + learning_rate*(d_W[j] + weight_decay*weights[j])
                    u_b[j] = momentum*u_b[j] + learning_rate*d_b[j]

                    weights[j] -= learning_rate*(d_W[j] + weight_decay*weights[j])
                    bias[j] -= learning_rate*d_b[j]

                d_W, d_b = reset_d_weights(weights, bias)

            if optimization_method == 'rmsprop':

                if epoch == 0 and i == 0:
                    v_W, v_b = reset_d_weights(weights, bias)

                for j in range(len(weights)):
                    v_W[j] = beta*v_W[j] + (1-beta)*d_W[j]**2
                    v_b[j] = beta*v_b[j] + (1-beta)*d_b[j]**2

                    weights[j] -= learning_rate*(d_W[j] + weight_decay*weights[j])/np.sqrt(v_W[j] + epsilon)
                    bias[j] -= learning_rate*d_b[j]/np.sqrt(v_b[j] + epsilon)

                d_W, d_b = reset_d_weights(weights, bias)

            if optimization_method == 'adam':

                if epoch == 0 and i == 0:
                    m_W, m_b = reset_d_weights(weights, bias)
                    v_W, v_b = reset_d_weights(weights, bias)
                
                for j in range(len(weights)):
                    m_W[j] = beta1*m_W[j] + (1-beta1)*d_W[j]
                    m_b[j] = beta1*m_b[j] + (1-beta1)*d_b[j]

                    v_W[j] = beta2*v_W[j] + (1-beta2)*(d_W[j]**2)
                    v_b[j] = beta2*v_b[j] + (1-beta2)*(d_b[j]**2)

                    m_W_hat = m_W[j]/(1-(beta1**step_cnt))
                    v_W_hat = v_W[j]/(1-(beta2**step_cnt))
                    m_b_hat = m_b[j]/(1-(beta1**step_cnt))
                    v_b_hat = v_b[j]/(1-(beta2**step_cnt))
                    
                    weights[j] -= learning_rate*((m_W_hat + weight_decay*weights[j])/(np.sqrt(v_W_hat) + epsilon))
                    bias[j] -= ((learning_rate*m_b_hat)/(np.sqrt(v_b_hat) + epsilon))

                d_W, d_b = reset_d_weights(weights, bias)
            
            if optimization_method == 'nadam':

                if epoch == 0 and i == 0:
                    m_W, m_b = reset_d_weights(weights, bias)
                    v_W, v_b = reset_d_weights(weights, bias)

                for j in range(len(weights)):
                    m_W[j] = beta1*m_W[j] + (1-beta1)*d_W[j]
                    m_b[j] = beta1*m_b[j] + (1-beta1)*d_b[j]

                    v_W[j] = beta2*v_W[j] + (1-beta2)*(d_W[j]**2)
                    v_b[j] = beta2*v_b[j] + (1-beta2)*(d_b[j]**2)

                    m_W_hat = m_W[j]/(1-(beta1**step_cnt))
                    v_W_hat = v_W[j]/(1-(beta2**step_cnt))
                    m_b_hat = m_b[j]/(1-(beta1**step_cnt))
                    v_b_hat = v_b[j]/(1-(beta2**step_cnt))

                    weights[j] -= (learning_rate*(((beta1*m_W_hat) + ((1-beta1)*d_W[j]/(1-(beta1**step_cnt)))) + weight_decay*weights[j]))/(np.sqrt(v_W_hat) + epsilon)
                    bias[j] -= (learning_rate*((beta1*m_b_hat) + ((1-beta1)*d_b[j]/(1-(beta1**step_cnt)))))/(np.sqrt(v_b_hat) + epsilon)

                d_W, d_b = reset_d_weights(weights, bias)

        if logging_train:
            y_pred = feedforward(X_data, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_data, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_data, y_pred)
            wandb.log({"train_accuracy": accuracy})
            wandb.log({"train_error": loss})
        
        if logging_val:
            y_pred = feedforward(X_val, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_val, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_val, y_pred)
            wandb.log({"validation_accuracy": accuracy})
            wandb.log({"validation_error": loss})
        
        if logging_test:
            y_pred = feedforward(X_test, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_test, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_test, y_pred)
            wandb.log({"test_accuracy": accuracy})
            wandb.log({"test_error": loss})

    return weights, bias