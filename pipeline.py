from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from utils.model import JEPA_Model, init_opt

import matplotlib.pyplot as plt
import os

import hydra
import wandb
import submitit_patch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/th3129/shared/DL24FA"
    # data_path = "/vast/yw4142/datasets/DL24FA"

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


def load_model(device):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    # model = MockModel()
    model = JEPA_Model(device=device)
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


def save_continuous_frames_with_metadata(batch, output_dir="sample_frames"):
    """
    Save continuous frames from one sample and print corresponding locations and actions.

    Args:
        batch: WallSample with states, locations, and actions.
        output_dir: Directory to save the frames.
    """
    # Extract the first sample from the batch
    states = batch.states[0]  # Shape: [frames, channels, height, width]
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

def train(model, probe_train_ds, optimizer, device):
    total_loss = 0.0
    model.train()

    for batch in tqdm(probe_train_ds):
        obs, location, actions = batch
        obs = obs.to(device)   # shape: (B, T, 2, H, W)
        actions = actions.to(device)  # shape: (B, T-1, 2)

        loss = model.compute_loss(obs, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        avg_loss = total_loss / len(probe_train_ds)
    return avg_loss

        
def evaluate(model, probe_val_ds, device):
    with torch.no_grad():
        val_loss = 0.0
        for batch in probe_val_ds['normal']:
            obs, location, actions = batch
            obs = obs.to(device)   # shape: (B, T, 2, H, W)
            actions = actions.to(device)  # shape: (B, T-1, 2)
            
            loss = model.compute_loss(obs, actions)

        val_loss += loss
        avg_val_loss = val_loss / len(probe_val_ds['normal'])
    return avg_val_loss


@hydra.main(config_path=".", config_name="config")
def main(cfg: OmegaConf):

    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        print(f"Saving everything in: {cfg['saved_folder']}")

    wandb.init(
        project="dl-final ",
        config=OmegaConf.to_container(cfg),
    )
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    print(f"Number of training samples: {len(probe_train_ds)}")
    print(f"Number of validating samples: {len(probe_val_ds)}")

    action_dim = cfg.get('action_dim', 0)
    model = load_model(device)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        model.observation_encoder,
        model.predictor,
        iterations_per_epoch=len(probe_train_ds),
        start_lr=cfg.start_lr,
        ref_lr=cfg.ref_lr,
        warmup=cfg.warmup_epochs,
        num_epochs=cfg.num_epochs,
        wd=cfg.weight_decay,
        final_wd=cfg.final_weight_decay,
        final_lr=cfg.final_lr,
        use_bfloat16=cfg.use_bfloat16,
        ipe_scale=cfg.ipe_scale
    )

    num_epochs = cfg.num_epochs
    for epoch in range(num_epochs):
        avg_loss = train(model, probe_train_ds, optimizer, device)
        avg_val_loss = evaluate(model, probe_val_ds, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss}, Validation Loss: {avg_val_loss}")
        wandb.log({"Train Loss": avg_loss, "Validation Loss": avg_val_loss, "Epoch": epoch+1})

    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    wandb.finish()

if __name__ == "__main__":
    main()
