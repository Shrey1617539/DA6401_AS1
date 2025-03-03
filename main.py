import argparse
import wandb  # Make sure to install this package: pip install wandb
import T1, T3

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation error"},
    "parameters": {
        "number of epochs": {"values": [5, 10]},
        "number of hidden layers": {"values": [3,4,5]},
        "size of every hidden layer": {"values": [32, 64, 128]},
        "learning rate": {"values": [1e-3, 1e-4]},
        "batch size": {"values": [16, 32, 64]},
    },
}



def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    X_train, y_train, X_test, y_test = T1.load_data()



    
    print("Experiment complete. Check your wandb dashboard for logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script that logs to Weights & Biases (wandb)")
    parser.add_argument('--wandb_entity', type=str, required=True, help='Your wandb entity (username or team name)')
    parser.add_argument('--wandb_project', type=str, required=True, help='Your wandb project name')
    args = parser.parse_args()
    
    main(args)

