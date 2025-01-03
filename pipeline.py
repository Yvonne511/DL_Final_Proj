from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import torch.nn as nn
from models import MockModel
import glob
from model import JEPA_Model, init_opt

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


def load_data(device, data_path):
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        batch_size=32,
        train=True,
    )

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        batch_size=64,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        batch_size=64,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        batch_size=64,
        train=False,
    )

    probe_val_wall_other_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall_other/val",
        probing=True,
        device=device,
        batch_size=64,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
        "wall_other": probe_val_wall_other_ds,
    }

    return train_ds, probe_train_ds, probe_val_ds


def load_expert_data(device, data_path):
    probe_train_expert_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_expert/train",
        probing=True,
        device=device,
        batch_size=64,
        train=True,
    )

    probe_val_expert_ds = {
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
        )
    }

    return probe_train_expert_ds, probe_val_expert_ds


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
    save_path = f"model_weights.pth"
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

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path=".", config_name="config")
def main(cfg: OmegaConf):

    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        print(f"Saving everything in: {cfg['saved_folder']}")
    run_name = '/'.join(cfg.saved_folder.split('/')[6:])
    """wandb.init(
        project="dl-final ",
        config=OmegaConf.to_container(cfg),
        name=run_name,
    )"""
    seed_everything(42)
    device = get_device()
    data_path = cfg.data_path
    train_ds, probe_train_ds, probe_val_ds = load_data(device, data_path)
    print(f"Number of training batches: {len(train_ds)}")
    print(f"Number of probe training batches: {len(probe_train_ds)}")
    print(f"Number of probe validating batches: {len(probe_val_ds)}")

    model_config = cfg.model
    action_dim = model_config.action_dim

    training_config = cfg.training
    num_epochs = training_config.epochs
    ipe = len(train_ds)
    ema = training_config.ema

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*training_config.ipe_scale)
                          for i in range(int(ipe*num_epochs*training_config.ipe_scale)+1))
                          
    model = JEPA_Model(device=device, 
                        action_dim=action_dim, 
                        momentum_scheduler=momentum_scheduler)

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
    for layer in model.modules():
        if isinstance(layer, nn.Linear):  # Adjust to target specific layers
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    # checkpoint_path = "model_checkpoint.pth"
    # start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
    sampling = True
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_f1_loss = 0.0
        total_var_loss = 0.0
        total_cov_loss = 0.0
        i = 0
        data_size = len(train_ds) if not sampling else 100
        for batch in tqdm(train_ds, desc=f"Training Epoch {epoch+1}", leave=False):
            if i > data_size:
                break
            obs = batch.states # [64, 17, 2, 65, 65]
            acts = batch.actions # [64, 17, 2]
            optimizer.zero_grad()
            loss, f1_loss, var_loss, cov_loss = model(obs, acts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            wd_scheduler.step()

            total_loss += loss.item()
            total_f1_loss += f1_loss.item()
            total_var_loss += var_loss.item()
            total_cov_loss += cov_loss.item()
            i += 1

        avg_loss = total_loss / data_size
        avg_f1_loss = total_f1_loss / data_size
        avg_var_loss = total_var_loss / data_size
        avg_cov_loss = total_cov_loss / data_size
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, F1 Loss: {avg_f1_loss:.4f}, "
          f"Var Loss: {avg_var_loss:.4f}, Cov Loss: {avg_cov_loss:.4f}")
        """wandb.log({
            "Train Loss": avg_loss,
            "F1 Loss": avg_f1_loss,
            "Var Loss": avg_var_loss,
            "Cov Loss": avg_cov_loss,
            "Epoch": epoch+1
        })"""

        # Evaluation on validation set
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0.0
        #     for batch in probe_val_ds['normal']:
        #         obs = batch.states
        #         acts = batch.actions
        #         loss = model(obs, acts)
        #         val_loss += loss.item()

        #     avg_val_loss = val_loss / len(probe_val_ds['normal'])
        #     print(f"Validation Loss: {avg_val_loss}")
        #     wandb.log({"Validation Loss": avg_val_loss, "Epoch": epoch+1})

        
        save_checkpoint(model, optimizer, epoch + 1, avg_loss)

        # if (epoch+1) % 10 == 0:
        #     avg_losses = evaluate_model(device, model, probe_train_ds, probe_val_ds)
        #     for probe_attr, loss in avg_losses.items():
        #         print(f"{probe_attr} loss: {loss}")
        #         wandb.log({f"Validation {probe_attr} Loss": loss})

    """wandb.finish()"""

if __name__ == "__main__":
    main()
