from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import torch.nn as nn
from models import MockModel
import glob
from utils.model import JEPA_Model, init_opt
from eval import evaluation

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import time


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device, cfg):
    data_path = "/scratch/qt2094/DL24FA"
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=cfg.training.batch_size
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

def save_checkpoint(model, optimizer, epoch, loss, cfg, start_time):
    # Use the Hydra-generated directory structure
    checkpoint_dir = f"{cfg.ckpt_base_path}/{start_time}"
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
    
    
@hydra.main(config_path=".", config_name="config")
def main(cfg: OmegaConf):
    seed = cfg.training.seed
    start_time = time.strftime("%Y-%m-%d/%H-%M-%S")

    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        print(f"Saving everything in: {cfg['saved_folder']}")

    wandb.init(
        project="dl-final ",
        config=OmegaConf.to_container(cfg),
    )
    device = get_device()
    train_ds, probe_train_ds, probe_val_ds = load_data(device, cfg)
    print(f"Number of training batches: {len(train_ds)}")
    print(f"Number of total batches: {len(train_ds) * cfg.training.epochs}")

    model_config = cfg.model
    action_dim = model_config.action_dim

    training_config = cfg.training
    num_epochs = training_config.epochs
    ipe = len(train_ds)
    ema = training_config.ema

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*training_config.ipe_scale)
                          for i in range(int(ipe*num_epochs*training_config.ipe_scale)+1))
                          
    model = JEPA_Model(model_name='vit_tiny', device=device, action_dim=action_dim, momentum_scheduler=momentum_scheduler)
    checkpoint_path = f"/scratch/qt2094/HW/DL_work/DL_Final_Proj/runs/2024-12-15/02-52-52/ckpt_1_1.477003.pth"
    

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
    #model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    #optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
    torch.manual_seed(seed)  # Set random seed for reproducibility
    for layer in model.modules():
        if isinstance(layer, nn.Linear):  # Adjust to target specific layers
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    # checkpoint_path = "model_checkpoint.pth"
    # start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_epoch = epoch + 1
        model.train()
        total_loss = 0.0
        total_f1_loss = 0.0
        total_var_loss = 0.0
        total_cov_loss = 0.0
        len_train_ds = len(train_ds)
        pbar = tqdm(train_ds, desc=f"Training Epoch {epoch+1}", leave=False)
        
        best_batch_loss = float('inf')
       

        for i, batch in enumerate(pbar):
            
            
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


            if i % cfg.training.eval_interval == 0:

                saving_begin = time.time()
                best_batch_loss = loss
                evaluation(cfg, model)
                save_checkpoint(model, optimizer, epoch + 1, best_batch_loss, cfg, start_time)
                saving_end = time.time()
                saving_time = saving_end - saving_begin
                print(f"Saving checkpoint took {saving_time:.2f} seconds")

            pbar.set_description(f"Epoch {current_epoch} - Loss: {loss:.4f}, "
                                  f"F1: {f1_loss:.4f}, "
                                  f"Var: {var_loss:.4f}, "
                                    f"Cov: {cov_loss:.4f}")
            wandb.log({
                "Batch Train Loss": loss,
                "Batch F1 Loss": f1_loss,
                "Batch Var Loss": var_loss,
                "Batch Cov Loss": cov_loss,
                "current learning rate": optimizer.param_groups[0]['lr'],
                "Epoch": current_epoch,
                "Batch": i+1
            })

        avg_loss = total_loss / len_train_ds
        avg_f1_loss = total_f1_loss / len_train_ds
        avg_var_loss = total_var_loss / len_train_ds
        avg_cov_loss = total_cov_loss / len_train_ds
        print(f"Epoch [{current_epoch}/{num_epochs}] - Loss: {avg_loss:.4f}, F1 Loss: {avg_f1_loss:.4f}, "
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
            save_checkpoint(model, optimizer, current_epoch, best_loss, cfg, start_time)


    wandb.finish()

    
if __name__ == "__main__":
    main()
