import torch
from brainsight.train import train_func_wandb
import wandb
import os
api_key = os.getenv('WANDB_API_KEY')
torch.manual_seed(4)

os.environ["WANDB_SILENT"] = "true"

# set configuration of Weights & Biases
my_config = {
    "method": "grid",
    "name": "sweep",
    "metric": {
        'goal': 'maximize',
        'name': 'accuracy_mean'
    },
    "parameters": {
        "subject": {"values": [[5, 6, 7, 8, 9, 10, 12]]},  # list of subjects
        "data": {"values": ["cst"]},
        "model_type": {"values": ["GRU"]},
        "hidden_dim": {"values": [3, 5, 10]},
        "num_layers": {"values": [5]},
        "batch_size": {"values": [64]},
        "window_size": {"values": [2048]},
        "step_size": {"values": [128]},
        "optimiser": {"values": ["Adam"]},
        "learning_rate": {"values": [1e-3]}
    }
}


def train(config=None):
    """ Set up sweep and run training with wandb. """

    with wandb.init(entity="bz21", project="Brainsight", config=config):

        config = wandb.config
        subject = config.subject
        data_type = config.data
        model_type = config.model_type
        data_window = config.window_size
        step_size = config.step_size
        optimiser = config.optimiser
        lr = config.learning_rate
        batch_size = config.batch_size
        hidden_dim = config.hidden_dim
        num_layers = config.num_layers

        if model_type == "GRU":
            wandb.run.name = "All_" + data_type + "_" + model_type + "_hidden" + \
                str(hidden_dim) + "_layers" + str(num_layers) + "_batch" + str(batch_size) + "_opt" + \
                str(optimiser) + "_lr" + str(lr) + "_win" + str(data_window) + "_step" + str(step_size)
        else:
            wandb.run.name = "All_" + data_type + "_" + model_type + "_batch" + \
                str(batch_size) + "_opt" + str(optimiser) + "_lr" + str(lr) + "_win" + str(data_window) + \
                "_step" + str(step_size)

        train_func_wandb(subject, data_type, model_type, data_window, step_size, optimiser, lr, batch_size,
                         hidden_dim, num_layers)


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=my_config, entity="bz21", project="Brainsight")
    wandb.agent(sweep_id, function=train)
