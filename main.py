import argparse
import wandb  
import T1, T3
import numpy as np
import json

def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)


    X_train, y_train, X_test, y_test = T1.load_data()
    X_train, y_train, X_val, y_val = T1.train_test_split(X_train, y_train, split_ratio=0.9)

    hidden_layers = get_config_value(wandb.config, args, "num_layers")
    initialisation = get_config_value(wandb.config, args, "weight_init")
    size_of_every_hidden_layer = get_config_value(wandb.config, args, "hidden_size")
    if isinstance(size_of_every_hidden_layer, str):
        size_of_every_hidden_layer = json.load(size_of_every_hidden_layer)

    if not isinstance(size_of_every_hidden_layer, list):
        size_of_every_hidden_layer = [size_of_every_hidden_layer for _ in range(hidden_layers)]

    weights, bias = T3.initialize_weights(
        input_size = X_train[0].shape[0] * X_train[0].shape[1],
        hidden_layers = hidden_layers,
        hidden_layer_size = size_of_every_hidden_layer,
        output_size = np.unique(y_train).shape[0],
        initialisation =initialisation
    )

    activation_function = [get_config_value(wandb.config, args, "activation") for _ in range(hidden_layers)]
    activation_function.append('softmax')
    epochs = get_config_value(wandb.config, args, 'epochs')
    learning_rate = get_config_value(wandb.config, args, 'learning_rate')
    beta = get_config_value(wandb.config, args, 'beta')
    beta1 = get_config_value(wandb.config, args, 'beta1')
    beta2 = get_config_value(wandb.config, args, 'beta2')
    weight_decay = get_config_value(wandb.config, args, 'weight_decay')
    momentum = get_config_value(wandb.config, args, 'momentum')
    batch_size = get_config_value(wandb.config, args, 'batch_size')
    optimization_method = get_config_value(wandb.config, args, 'optimizer')
    epsilon = get_config_value(wandb.config, args, 'epsilon')
    loss_type = get_config_value(wandb.config, args, 'loss')

    run_name = f"hl_{hidden_layers}_bs_{batch_size}_ac_{activation_function[0]}_lr_{learning_rate}_opt_{optimization_method}"
    run_name += f"_wd_{weight_decay}_mom_{momentum}_epochs_{epochs}"
    wandb.run.name = run_name
    weights, bias = T3.gradient_descent(
        X_data=X_train, 
        Y_data=y_train, 
        weights=weights, 
        bias=bias, 
        epochs=epochs, 
        activation_function=activation_function, 
        learning_rate=learning_rate, 
        beta=beta,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        momentum=momentum,
        batch_size=batch_size, 
        optimization_method=optimization_method,
        epsilon=epsilon,
        loss_type=loss_type,
        X_val=X_val, 
        Y_val=y_val, 
        X_test=X_test, 
        Y_test=y_test, 
        logging_train=True, 
        logging_val=True,
        logging_test=False
    )
    print("Experiment complete. Check your wandb dashboard for logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script that logs to Weights & Biases (wandb)"
    )
    parser.add_argument(
        '-we',
        '--wandb_entity',
        type=str,
        default='me21b138-indian-institute-of-technology-madras',
        help='Wandb Entity used to track experiments in the Weights & Biases dashboard'
    )
    parser.add_argument(
        '-wp',
        '--wandb_project',
        type=str,
        default='my-awesome-project',
        help='Project name used to track experiments in Weights & Biases dashboard'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='fashion_mnist',
        help='choices: ["mnist", "fashion_mnist"]'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs to train neural network.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=4,
        help='Batch size used to train neural network.'
    )
    parser.add_argument(
        '-l',
        '--loss',
        type=str,
        default='cross_entropy',
        help='choices: ["mean_squared_error", "cross_entropy"]'
    )
    parser.add_argument(
        '-o',
        '--optimizer',
        type=str,
        default='adam',
        help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate used to optimize model parameters'
    )
    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        default=0.5,
        help='Momentum used by momentum and nag optimizers.'
    )
    parser.add_argument(
        '-beta',
        '--beta',
        type=float,
        default=0.5,
        help='Beta used by rmsprop optimizer'
    )
    parser.add_argument(
        '-beta1',
        '--beta1',
        type=float,
        default=0.9,
        help='Beta1 used by adam and nadam optimizers'
    )
    parser.add_argument(
        '-beta2',
        '--beta2',
        type=float,
        default=0.999,
        help='Beta2 used by adam and nadam optimizers'
    )
    parser.add_argument(
        '-eps',
        '--epsilon',
        type=float,
        default=1e-6,
        help='Epsilon used by optimizers.'
    )
    parser.add_argument(
        '-w_d',
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay used by optimizers.'
    )
    parser.add_argument(
        '-w_i',
        '--weight_init',
        type=str,
        default="random",
        help='choices: ["random", "Xavier"]'
    )
    parser.add_argument(
        '-nhl',
        '--num_layers',
        type=int,
        default=3,
        help='Number of hidden layers used in feedforward neural network.'
    )
    parser.add_argument(
        '-sz',
        '--hidden_size',
        type=str,
        default='128',
        help='Number of hidden neurons in a feedforward layer.'
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        default='sigmoid',
        help='choices: ["identity", "sigmoid", "tanh", "ReLU"]'
    )
    # Use parse_known_args to ignore extra arguments injected by wandb sweep agent
    args = parser.parse_args()
    main(args)


