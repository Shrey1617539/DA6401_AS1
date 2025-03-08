import wandb
import yaml

# Initialize sweep with core backend
with open("sweep.yaml", "r") as file:
    sweep_configuration = yaml.safe_load(file)
wandb.require("core")
sweep_id = wandb.sweep(
    sweep_configuration,
    entity='me21b138-indian-institute-of-technology-madras',
    project='my-awesome-project'
)
wandb.agent(sweep_id)
