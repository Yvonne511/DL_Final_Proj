from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from utils.model import JEPA_Model, init_opt


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


def load_model(device, action_dim):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    # model = MockModel()
    model = JEPA_Model(device=device, action_dim=action_dim)
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
    print(len(probe_train_ds))
    print(len(probe_val_ds))

    # for batch in probe_train_ds:
    #     print(batch[0].shape)
    #     print(batch[1])
    #     break
    action_dim = cfg.get('action_dim', 0)
    model = load_model(device, action_dim=action_dim)
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
        model.train()
        total_loss = 0.0
        for batch in probe_train_ds:
            current_observation, action, future_observation = batch
            current_observation = current_observation.to(device)
            action = action.to(device)
            future_observation = future_observation.to(device)

            optimizer.zero_grad()
            loss = model(current_observation, future_observation, action=action)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(probe_train_ds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")
        wandb.log({"Train Loss": avg_loss, "Epoch": epoch+1})

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in probe_val_ds['normal']:
                current_observation, action, future_observation = batch
                current_observation = current_observation.to(device)
                action = action.to(device)
                future_observation = future_observation.to(device)

                loss = model(current_observation, future_observation, action=action)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(probe_val_ds['normal'])
            print(f"Validation Loss: {avg_val_loss}")
            wandb.log({"Validation Loss": avg_val_loss, "Epoch": epoch+1})

    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    wandb.finish()

if __name__ == "__main__":
    main()
