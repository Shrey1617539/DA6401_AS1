import argparse
import wandb  # Make sure to install this package: pip install wandb
import T1, T3
import numpy as np
import functools

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation_error"},
    "parameters": {
        "number_of_epochs": {"values": [5, 10]},
        "number_of_hidden_layers": {"values": [3, 4, 5]},
        "size_of_every_hidden_layer": {"values": [32, 64, 128]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "batch_size": {"values": [16, 32, 64]},
        "initialisation": {"values": ["random"]},
        "activation_function": {"values": ["sigmoid"]},
    },
}

def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    X_train, y_train, X_test, y_test = T1.load_data()
    X_train, y_train, X_val, y_val = T1.train_test_split(X_train, y_train, split_ratio=0.9)
    weights, bias = T3.initialize_weights(X_train[0].flatten().shape[0], wandb.config.number_of_hidden_layers, wandb.config.size_of_every_hidden_layer, np.unique(y_train).shape[0], wandb.config.initialisation)
    weights, bias = T3.gradient_descent(X_train, y_train, weights, bias,  wandb.config.number_of_epochs, activation_function= [wandb.config.activation_function for i in range(wandb.config.number_of_hidden_layers)], learning_rate = wandb.config.learning_rate, batch_size = wandb.config.batch_size)
    y_val_pred = []
    for i in range(X_val.shape[0]):
        y_val_pred.append(T3.feedforward(X_val[i], weights, bias, activation_function = [wandb.config.activation_function for i in range(wandb.config.number_of_hidden_layers)])[0])
    loss = T3.loss_calculations(y_val, y_val_pred)
    wandb.log({"validation_error": loss})

    print("Experiment complete. Check your wandb dashboard for logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script that logs to Weights & Biases (wandb)")
    parser.add_argument('--wandb_entity', type=str, required=True, help='Your wandb entity (username or team name)')
    parser.add_argument('--wandb_project', type=str, required=True, help='Your wandb project name')
    args = parser.parse_args()
    sweep_id = wandb.sweep(sweep_configuration, entity=args.wandb_entity, project=args.wandb_project)
    wandb.agent(sweep_id, function=functools.partial(main, args), count = 1)
    # main(args)

