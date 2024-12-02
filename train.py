from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/vast/yw4142/datasets/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    model = MockModel()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

import os
import hydra
import wandb
import submitit_patch
from omegaconf import OmegaConf, open_dict

@hydra.main(config_path=".", config_name="config")
def main(cfg: OmegaConf):

    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        print(f"Saving everything in: {cfg['saved_folder']}")
    model_name = cfg["saved_folder"].split("outputs/")[-1]
    # model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"
    total_epochs = cfg.training.epochs
    epoch = 0
    wandb_run = wandb.init(
        project="dl-final ",
        config=OmegaConf.to_container(cfg),
    )

    with open_dict(cfg):
        cfg.wandb_run_id = wandb_run.id
        cfg.wandb_run_name = model_name  # Save the run name for tracking
        print(f"W&B run ID: {cfg.wandb_run_id}")
        print(f"W&B run name: {cfg.wandb_run_name}")

    wandb.run.name = model_name

    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    print(len(probe_train_ds))
    print(len(probe_val_ds))

    # for batch in probe_train_ds:
    #     print(batch[0].shape)
    #     print(batch[1])
    #     break
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    wandb.finish()

if __name__ == "__main__":
    main()
