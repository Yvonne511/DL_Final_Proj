from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import torch.nn as nn
from models import MockModel
import glob
from utils.model import JEPA_Model, init_opt
import pprint
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import time


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device, batch_size):
    data_path = "/scratch/qt2094/DL24FA"
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size= batch_size
    )
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

    return train_ds, probe_train_ds, probe_val_ds


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

def save_checkpoint(model, optimizer, epoch, loss, start_time):
    # Use the Hydra-generated directory structure
    saved_folder = saved_folder = "/scratch/qt2094/HW/DL_work/DL_Final_Proj/runs"
    checkpoint_dir = f"{saved_folder}/{start_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    
    # Save the current checkpoint
    save_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}_{loss:.6f}.pth")
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
    
"""def create_sweep_cfg():
    sweep_cfg = {
        "method": "bayes",
    }
    metric = {
            "name": "Batch eval Loss",
            "goal": "minimize"
    }
    sweep_cfg["metric"] = metric
    sweep_cfg["parameters"] = {
            # Model parameters
            "model_embed_dim": {
                "values": [96, 128, 192]
            },
            "model_depth": {
                "values": [6, 8, 12]
            },
            "model_num_heads": {
                "values": [3, 6, 9]
            },
            "model_patch_size": {
                "values": [5, 16]
            },
            "model_mlp_ratio": {
                "values": [4, 6]
            },
            "model_pred_depth": {
                "values": [3, 4, 6]
            },
            "model_pred_num_heads": {
                "values": [4, 6]
            },

            # Training parameters
            "training_batch_size": {
                "values": [32, 64, 128]
            },
            "training_start_lr": {
                "distribution": "log_uniform",
                "min": -7,  # 1e-7
                "max": -4   # 1e-4
            },
            "training_ref_lr": {
                "distribution": "log_uniform",
                "min": -5,  # 1e-5
                "max": -2   # 1e-2
            },
            "training_final_lr": {
                "distribution": "log_uniform",
                "min": -8,  # 1e-8
                "max": -5   # 1e-5
            },
            "training_weight_decay": {
                "distribution": "log_uniform",
                "min": -4.39794,  # log10(0.00004)
                "max": -1.39794   # log10(0.04)
            },
            "training_warmup_epochs": {
                "values": [5, 10]
            },
            "training_ema": {
                "values": [[0.996, 1.0], [0.998, 1.0]]
            },
            "training_use_bfloat16": {
                "values": [True, False]
            },
            "training_ipe_scale": {
                "values": [1.0, 2.0, 4.0]
            },
            
            # Hyperband parameters
            "bracket": {
                "values": [0, 1, 2, 3, 4]
            },
            "rung": {
                "values": [0, 1, 2, 3, 4]
            }
    }
    early_terminate = {
        "type": "hyperband",
        "min_iter": 1,
        "max_iter": 81,
        "eta": 3
    }
    sweep_cfg["early_terminate"] = early_terminate
    
    return sweep_cfg

sweep_cfg = create_sweep_cfg()
"""
sweep_cfg = OmegaConf.load("./sweep_config.yaml")
sweep_cfg = OmegaConf.to_container(sweep_cfg, resolve=True)
def sweep_main(config = sweep_cfg):
    start_time = time.strftime("%Y-%m-%d/%H-%M-%S")
    # convert to dict to pass to wandb
    #sweep_cfg = OmegaConf.to_container(sweep_cfg, resolve=True)
    
    with wandb.init(
        project="dl-final-tuning",
        config=sweep_cfg,
    ):
        config = wandb.config
        batch_size = config.training_batch_size
        #print('=========================================\n',config.early_terminate)
        max_iter = config.early_terminate['max_iter']  # 81
        eta = config.early_terminate['eta']  # 3
        
        s_max = int(np.log(max_iter) / np.log(eta))  # Should be 4
        # Get current bracket and rung from wandb
        current_bracket = wandb.config.get('bracket', s_max)
        current_rung = wandb.config.get('rung', 0)
        
        # Calculate epochs for current rung
        epochs_per_rung = max_iter // (eta ** (s_max - current_bracket + current_rung))
        saved_folder = "/scratch/qt2094/HW/DL_work/DL_Final_Proj/runs"
        print(f"Saving everything in: {saved_folder}")

        
        device = get_device()
        train_ds, _, _ = load_data(device, batch_size)
        #print(f"Number of training batches: {len(train_ds)}")
        #print(f"Number of total batches: {len(train_ds) * epochs_per_rung}")

        action_dim = 2

        #num_epochs = training_config.epochs
        ipe = len(train_ds)
        ema = config.training_ema

        momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*epochs_per_rung*config.training_ipe_scale)
                            for i in range(int(ipe*epochs_per_rung*config.training_ipe_scale)+1))                                             
        
        model = JEPA_Model(model_name='vit_custom',
                        embed_dim=config.model_embed_dim,
                        encoder_depth=config.model_depth,
                        encoder_num_heads=config.model_num_heads,
                        pred_depth=config.model_pred_depth,
                        pred_num_heads=config.model_pred_num_heads,
                        patch_size=config.model_patch_size,
                        device=device,
                        action_dim=action_dim, 
                        mlp_ratio=config.model_mlp_ratio,
                        momentum_scheduler=momentum_scheduler)
        
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
                model.observation_encoder,
                model.predictor,
                iterations_per_epoch=ipe,
                start_lr=config.training_start_lr,
                ref_lr=config.training_ref_lr,
                warmup=config.training_warmup_epochs,
                num_epochs=epochs_per_rung,  # Use epochs_per_rung instead of fixed value
                wd=config.training_weight_decay,
                final_lr=config.training_final_lr,
                use_bfloat16=config.training_use_bfloat16,
                ipe_scale=config.training_ipe_scale
            )
        
        for layer in model.modules():
            if isinstance(layer, nn.Linear):  # Adjust to target specific layers
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    
        best_loss = float('inf')

        for epoch in range(epochs_per_rung):
            current_epoch = epoch + 1
            model.train()
            total_loss = 0.0
            total_f1_loss = 0.0
            total_var_loss = 0.0
            total_cov_loss = 0.0
            len_train_ds = len(train_ds)
            pbar = tqdm(train_ds, desc=f"Training Epoch {epoch+1}", leave=False)
            
            best_batch_loss = float('inf')
        
            total_batch = 0
            for i, batch in enumerate(pbar):
                total_batch += 1
                
                # print(f"Batch states shape: {batch.states.shape}")  # [64, 17, 2, 65, 65]
                # print(f"Batch locations shape: {batch.locations.shape}") # [64, 17, 2]
                # print(f"Batch actions shape: {batch.actions.shape}") # [64, 16, 2]
                # save_continuous_frames_with_metadata(batch)
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


                #if total_batch % 2 == 0:
                    
                #    saving_begin = time.time()
                #    best_batch_loss = loss
                #    save_checkpoint(model, optimizer, epoch + 1, best_batch_loss, cfg, start_time)
                #    saving_end = time.time()
                #    saving_time = saving_end - saving_begin
                #    print(f"Saving checkpoint took {saving_time:.2f} seconds")

                pbar.set_description(f"Epoch {current_epoch} - Loss: {loss:.4f}, "
                                    f"F1: {f1_loss:.4f}, "
                                    f"Var: {var_loss:.4f}, "
                                        f"Cov: {cov_loss:.4f}")
                wandb.log({
                    "Batch Train Loss": loss,
                    "Batch F1 Loss": f1_loss,
                    "Batch Var Loss": var_loss,
                    "Batch Cov Loss": cov_loss,
                    "Epoch": current_epoch,
                    "Batch": i+1
                })

            avg_loss = total_loss / len_train_ds
            avg_f1_loss = total_f1_loss / len_train_ds
            avg_var_loss = total_var_loss / len_train_ds
            avg_cov_loss = total_cov_loss / len_train_ds
            print(f"Epoch [{current_epoch}/{epochs_per_rung}] - Loss: {avg_loss:.4f}, F1 Loss: {avg_f1_loss:.4f}, "
            f"Var Loss: {avg_var_loss:.4f}, Cov Loss: {avg_cov_loss:.4f}")
            wandb.log({
                "Epoch Summary": {
                    "Train Loss": avg_loss,
                    "F1 Loss": avg_f1_loss,
                    "Var Loss": avg_var_loss,
                    "Cov Loss": avg_cov_loss,
                    "Epoch": current_epoch
                }
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, current_epoch, best_loss, start_time)


    return best_loss

if __name__ == "__main__":
    #sweep_cfg = create_sweep_cfg()

    # convert to dict to pass to wandb
    #sweep_cfg = OmegaConf.to_container(sweep_cfg, resolve=True)
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep = sweep_cfg, project="dl-final-sweep")
    #sweep_main()
    wandb.agent(sweep_id, function=sweep_main, count = 81)