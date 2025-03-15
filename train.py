import argparse
import wandb  
import data_loading_function, model_training_function
import numpy as np

def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    X_train, y_train, X_test, y_test = data_loading_function.load_data()

    if args.plot_image == True:
        data_loading_function.plot_images(X_train=X_train, y_train=y_train)

    X_train, y_train, X_val, y_val = model_training_function.train_test_split(X_train, y_train, split_ratio=0.9)

    if args.load_model == "":

        hidden_layers = get_config_value(wandb.config, args, "num_layers")
        initialisation = get_config_value(wandb.config, args, "weight_init")
        size_of_every_hidden_layer = get_config_value(wandb.config, args, "hidden_size")
        if not isinstance(size_of_every_hidden_layer, list):
            size_of_every_hidden_layer = [size_of_every_hidden_layer for _ in range(hidden_layers)]

        weights, bias = model_training_function.initialize_weights(
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

        weights, bias = model_training_function.gradient_descent(
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

        if args.model_name == "":
            model_name = run_name+".npz"
        else:
            model_name = args.model_name
        np.savez(model_name, *weights, *bias, np.array(activation_function, dtype=object))

    else:
        loaded_model = np.load(args.load_model, allow_pickle=True)
        num_layers = (len(loaded_model.files) - 1)//2

        weights = [loaded_model[f'arr_{i}'] for i in range(num_layers)]
        bias = [loaded_model[f'arr_{i}'] for i in range(num_layers, 2*num_layers)]
        activation_function = loaded_model[f'arr_{2*num_layers}'].tolist()
    
    if args.evaluate_training == True:
        Y_pred_train = model_training_function.feedforward(X_train, weights=weights, bias=bias, activation_function=activation_function)[0]
        train_accuracy = model_training_function.accuracy_calculations(y_train, Y_pred_train)
        print('Train Accuracy : ', train_accuracy)
        model_training_function.confusion_matrix_plot(y_true=y_train, y_pred=Y_pred_train,title="Confusion Matrix Train")      

    if args.evaluate_validation == True:
        Y_pred_val = model_training_function.feedforward(X_val, weights=weights, bias=bias, activation_function=activation_function)[0]
        val_accuracy = model_training_function.accuracy_calculations(y_val, Y_pred_val)
        print('Validation Accuracy : ', val_accuracy)
        model_training_function.confusion_matrix_plot(y_true=y_val, y_pred=Y_pred_val,title="Confusion Matrix Validation")

    if args.evaluate_test == True:
        Y_pred_test = model_training_function.feedforward(X_test, weights=weights, bias=bias, activation_function=activation_function)[0]
        test_accuracy = model_training_function.accuracy_calculations(y_test, Y_pred_test)
        print('Test Accuracy : ', test_accuracy)
        model_training_function.confusion_matrix_plot(y_true=y_test, y_pred=Y_pred_test,title="Confusion Matrix Test") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script that return model weights"
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
        default=15,
        help='Number of epochs to train neural network.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=32,
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
        default=0.00045380889722810544,
        help='Learning rate used to optimize model parameters'
    )
    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        default=0.930262729134148,
        help='Momentum used by momentum and nag optimizers.'
    )
    parser.add_argument(
        '-beta',
        '--beta',
        type=float,
        default=0.01715447631841027,
        help='Beta used by rmsprop optimizer'
    )
    parser.add_argument(
        '-beta1',
        '--beta1',
        type=float,
        default=0.9907680279134712,
        help='Beta1 used by adam and nadam optimizers'
    )
    parser.add_argument(
        '-beta2',
        '--beta2',
        type=float,
        default=0.9969324668725889,
        help='Beta2 used by adam and nadam optimizers'
    )
    parser.add_argument(
        '-eps',
        '--epsilon',
        type=float,
        default=0.00000876236726267181,
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
        default="Xavier",
        help='choices: ["random", "Xavier"]'
    )
    parser.add_argument(
        '-nhl',
        '--num_layers',
        type=int,
        default=4,
        help='Number of hidden layers used in feedforward neural network.'
    )
    parser.add_argument(
        '-sz',
        '--hidden_size',
        type=int,
        default=128,
        help='Number of hidden neurons in a feedforward layer.'
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        default="tanh",
        help='choices: ["identity", "sigmoid", "tanh", "ReLU"]'
    )
    parser.add_argument(
        '-lm',
        '--load_model',
        type=str,
        default="",
        help='give name of you model you want to load here. e.g. "model.npz"'
    )
    parser.add_argument(
        '-pi',
        '--plot_image',
        type=bool,
        default=False,
        help="If you want to visualize every class of the training data, set True."
    )
    parser.add_argument(
        '-mn',
        '--model_name',
        type = str,
        default="",
        help='in what name you want to store the model. e.g "model.npz"'
    )
    parser.add_argument(
        '-etr',
        '--evaluate_training',
        type=bool,
        default=False,
        help='If you want to see evaluation on training data with confusion matrix, set True.'
    )
    parser.add_argument(
        '-eva',
        '--evaluate_validation',
        type=bool,
        default=False,
        help='If you want to see evaluation on validation data with confusion matrix, set True.'
    )
    parser.add_argument(
        '-ete',
        '--evaluate_test',
        type=bool,
        default=False,
        help='If you want to see evaluation on test data with confusion matrix, set True.'
    )
    # Use parse_known_args to ignore extra arguments injected by wandb sweep agent
    args = parser.parse_args()
    main(args)