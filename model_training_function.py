import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# Break given data into train and test split
def train_test_split(X_train, y_train, split_ratio=0.9, seed = 42):
    np.random.seed(seed)
    split = int(X_train.shape[0]*split_ratio)

    # randomize the data with cerain seed
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]
    return X_train[:split], y_train[:split], X_train[split:], y_train[split:]

# Different activation function's formulation 
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
        pos_mask = (a>0)
        result = np.zeros(a.shape)
        result[pos_mask] = a[pos_mask]
        
        return result
    
    if activation_function == 'identity':
        return a
    
    if activation_function == 'softmax':
        a = np.exp(a - np.max(a, axis=1, keepdims=True))
        a /= np.sum(a, axis=1, keepdims=True)
        return a

# gradients of different activation function
def gradient_activation_function(a, activation_function):
    if activation_function == 'sigmoid':
        z = activation_function_cal(a, activation_function=activation_function)
        return z*(1-z)
    
    if activation_function == 'tanh':
        z = activation_function_cal(a, activation_function=activation_function)
        return (1 - (z**2))
    
    if activation_function == 'ReLU':
        pos_mask = (a>0)
        result = np.zeros(a.shape)
        result1 = np.ones(a.shape)
        result[pos_mask] = result1[pos_mask]

        return result
    
    if activation_function == 'identity':
        return np.ones(a.shape)

# Feed Forward in the neural network
def feedforward(x, weights, bias, activation_function):
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],1)
    a = []
    h = []

    # Goes through all network but the last
    for layer in range(len(weights)-1):
        x = bias[layer] + ( weights[layer] @ x )
        a.append(x)
        x = activation_function_cal(x, activation_function[layer])
        h.append(x)

    # feed forward in last layer
    x = bias[-1] + (weights[-1] @ x)
    a.append(x)
    x = activation_function_cal(x, activation_function[-1])
    h.append(x)

    # return output, a, h
    return x, a, h
    
# Loss calculations for different type of loss
def loss_calculations(y, y_pred, loss_type = 'cross_entropy'):
    if type(y) == list:
        y = np.array(y)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if loss_type == 'cross_entropy':
        loss = -1*np.log(y_pred[np.arange(y_pred.shape[0]), y, 0]+1e-6)
        loss = np.average(loss)
    if loss_type == 'mean_squared_error' or loss_type == 'mse':
        y_array = np.zeros(y_pred.shape)
        y_array[np.arange(y_pred.shape[0]), y] = 1
        loss = np.sum((y_array - y_pred)**2, axis = 1)
        loss = np.mean(loss)
    return loss

# Getting accuracy with the help of actual and predicted values
def accuracy_calculations(y, y_pred):
    if type(y) == list:
        y = np.array(y)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    y_pred = np.argmax(y_pred, axis = 1)
    y_pred = y_pred.flatten()
    return np.sum(y_pred == y)/y.shape[0]

# Back propagation for batch and get descent value
def backpropagation(X, Y, weights, bias, activation_function, loss_type = 'cross_entropy'):

    d_W = []
    d_b = []

    # Getting values with feedforward
    y_pred, a, h = feedforward(X, weights = weights, bias= bias, activation_function= activation_function)

    e_y = np.zeros(y_pred.shape)
    e_y[np.arange(e_y.shape[0]),Y] = 1

    # getting grad_al for cross entropy
    if loss_type == 'cross_entropy':
        grad_al = (y_pred - e_y).reshape(e_y.shape[0], e_y.shape[1], 1)

    # getting grad_al for mean-squared-error
    if loss_type == 'mean_squared_error' or loss_type == 'mse':
        sum_c = (y_pred - e_y)*y_pred
        sum_c = np.sum(sum_c, axis = 1)
        sum_c = sum_c[:, np.newaxis, :]
        grad_al = 2*y_pred*(y_pred - e_y - sum_c).reshape(e_y.shape[0], e_y.shape[1], 1)
    grad_hl = 0

    # going through reversed order with help of chain rule
    for layer in reversed(range(1, len(weights))):
        d_W.append(np.matmul(grad_al, h[layer - 1].transpose(0,2,1)))
        d_b.append(grad_al)

        grad_hl = np.matmul(weights[layer].T, grad_al)
        grad_al = grad_hl * gradient_activation_function(a[layer - 1], activation_function[layer - 1])
    
    d_W.append(np.matmul(grad_al, X.reshape(X.shape[0], X.shape[1]*X.shape[2],1).transpose(0,2,1)))
    d_b.append(grad_al)

    d_W = list(reversed(d_W))
    d_b = list(reversed(d_b))
    
    # returning the list of the gradients
    return d_W, d_b
    
# reset inputs to zeros
def reset_d_weights(weights, bias):
    d_W = []
    d_b = []

    for i in range(len(weights)):
        d_W.append(np.zeros(weights[i].shape))
        d_b.append(np.zeros(bias[i].shape))
    
    return d_W, d_b

# Initialize weights and biases according to initialisation method.
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

# Gradient descent for weights and biases
def gradient_descent(X_data, Y_data, weights, bias, epochs, activation_function , 
                     learning_rate = 0.01, beta = 0.5, beta1 = 0.5, beta2 = 0.5, weight_decay = 0,
                     momentum = 0.5, batch_size = None, optimization_method = None,
                     epsilon = 0.000001, loss_type = 'cross_entropy', X_val = None, Y_val = None, X_test = None, 
                     Y_test = None, logging_train = False, logging_val = False, logging_test = False):
    
    # assign value of batch size if not 
    if batch_size == None:
        batch_size = X_data.shape[0]
    
    if optimization_method == 'sgd':
        batch_size = 1

    step_cnt = 0
    for epoch in range(epochs):

        # Create the randomize batch for every epoch
        indices = np.random.permutation(X_data.shape[0])
        X_data = X_data[indices]
        Y_data = Y_data[indices]

        d_W, d_b = reset_d_weights(weights, bias)

        for i in range(0, X_data.shape[0], batch_size):

            step_cnt += 1
            X = X_data[i:min(X_data.shape[0], i+batch_size)]
            Y = Y_data[i:min(X_data.shape[0], i+batch_size)]

            # If optimizers are nag then help to look ahaed gradients
            if optimization_method == 'nag':

                if epoch == 0 and i == 0:
                    u_W, u_b = reset_d_weights(weights, bias)
                
                for j in range(len(weights)):
                    weights[j] -= momentum*u_W[j]
                    bias[j] -= momentum*u_b[j]

            # getting gredients with backpropagation
            d_W_part, d_b_part = backpropagation(X, Y, weights = weights, bias = bias, activation_function = activation_function, loss_type=loss_type)

            for j in range(len(weights)):

                d_W[j] += np.mean(d_W_part[j], axis = 0)
                d_b[j] += np.mean(d_b_part[j], axis = 0)

            # Update rules for diffenent optimizers
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

        # Logging train accuracy and loss to wandb
        if logging_train:
            y_pred = feedforward(X_data, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_data, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_data, y_pred)
            wandb.log({"train_accuracy": accuracy})
            wandb.log({"train_error": loss})
        
        # Logging Validation accuracy and loss to wandb
        if logging_val:
            y_pred = feedforward(X_val, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_val, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_val, y_pred)
            wandb.log({"validation_accuracy": accuracy})
            wandb.log({"validation_error": loss})
        
        # Logging test accuracy and loss to wandb
        if logging_test:
            y_pred = feedforward(X_test, weights, bias, activation_function)[0]
            loss = loss_calculations(Y_test, y_pred, loss_type=loss_type)
            accuracy = accuracy_calculations(Y_test, y_pred)
            wandb.log({"test_accuracy": accuracy})
            wandb.log({"test_error": loss})

    return weights, bias

# PLotting confusion matrix
def confusion_matrix_plot(y_true, y_pred, class_names=None, title="Confusion Matrix", cmap="Blues"):
    if y_true.shape != y_pred.shape:
        y_pred = np.argmax(y_pred, axis=1).squeeze()

    classes = np.unique(y_true)
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for t, p in zip(y_true, y_pred):
        if p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
    
    if class_names == None:
        class_names = [f'class {i}' for i in range(len(np.unique(y_true)))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plot_filename = title+".png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()