from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from utils.model import JEPA_Model, init_opt

import matplotlib.pyplot as plt
import os

from tqdm import tqdm


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device, data_path):
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

    probe_val_wall_other_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall_other/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
        "wall_other": probe_val_wall_other_ds,
    }

    return probe_train_ds, probe_val_ds


def load_expert_data(device, data_path):
    probe_train_expert_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_expert/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_expert_ds = {
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            train=False,
        )
    }

    return probe_train_expert_ds, probe_val_expert_ds


def load_model(device, action_dim):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    model = MockModel()
    # model = JEPA_Model(device=device, action_dim=action_dim)
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
    return avg_losses


# if __name__ == "__main__":
#     device = get_device()
#     probe_train_ds, probe_val_ds = load_data(device)
#     model = load_model()
#     evaluate_model(device, model, probe_train_ds, probe_val_ds)

import os
import hydra
import wandb
import submitit_patch
from omegaconf import OmegaConf, open_dict

def save_continuous_frames_with_metadata(batch, output_dir="sample_frames"):
    """
    Save continuous frames from one sample and print corresponding locations and actions.

    Args:
        batch: WallSample with states, locations, and actions.
        output_dir: Directory to save the frames.
    """
    # Extract the first sample from the batch
    states = batch.states[0]  # Shape: [channels, frames, height, width]
    locations = batch.locations[0]  # Shape: [frames, 2]
    actions = batch.actions[0]  # Shape: [frames - 1, 2]
    frames, channels, height, width = states.shape

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save frames
    for f in range(frames):
        # Select the first channel for visualization
        frame = states[f, 1, :, :].squeeze().cpu().numpy()  # Shape: [65, 65]

        plt.imshow(frame, cmap="gray")
        plt.title(f"Sample 0, Frame {f}")
        plt.axis("off")
        plt.savefig(f"{output_dir}/frame_{f}.png")
        plt.close()

    print(f"Saved frames to {output_dir}")

    # Print corresponding locations and actions
    print("Locations:")
    print(locations.cpu().numpy())

    print("\nActions:")
    print(actions.cpu().numpy())

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    save_path = f"checkpoint_{epoch}.pth"
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, save_path="checkpoint.pth"):
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Checkpoint loaded from {save_path} at epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {save_path}")
        return 0, None

@hydra.main(config_path=".", config_name="config")
def main(cfg: OmegaConf):

    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        print(f"Saving everything in: {cfg['saved_folder']}")

    # wandb.init(
    #     project="dl-final ",
    #     config=OmegaConf.to_container(cfg),
    # )
    device = get_device()
    data_path = cfg.data_path
    probe_train_ds, probe_val_ds = load_data(device, data_path)
    print(f"Number of training batches: {len(probe_train_ds)}")
    print(f"Number of validating batches: {len(probe_val_ds)}")

    model_config = cfg.model
    action_dim = model_config.action_dim

    training_config = cfg.training
    num_epochs = training_config.epochs
    ipe = len(probe_train_ds)
    ema = training_config.ema

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*training_config.ipe_scale)
                          for i in range(int(ipe*num_epochs*training_config.ipe_scale)+1))
                          
    model = JEPA_Model(device=device, action_dim=action_dim, momentum_scheduler=momentum_scheduler)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
            model.observation_encoder,
            model.predictor,
            iterations_per_epoch=ipe,
            start_lr=training_config.start_lr,
            ref_lr=training_config.ref_lr,
            warmup=training_config.warmup_epochs,
            num_epochs=num_epochs,
            wd=training_config.weight_decay,
            final_wd=training_config.final_weight_decay,
            final_lr=training_config.final_lr,
            use_bfloat16=training_config.use_bfloat16,
            ipe_scale=training_config.ipe_scale
        )

    checkpoint_path = "/vast/yw4142/checkpoints/dl_final/outputs/2024-12-15/00-27-17/0/checkpoint_40.pth"
    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    model.eval()
    # evaluate_model(device, model, probe_train_ds, probe_val_ds)

    probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device, data_path)
    evaluate_model(device, model, probe_train_expert_ds, probe_val_expert_ds)

        # for probe_attr, loss in avg_losses.items():
        #     print(f"{probe_attr} loss: {loss}")
        #     wandb.log({f"Validation {probe_attr} Loss": loss})

    # wandb.finish()

if __name__ == "__main__":
    main()