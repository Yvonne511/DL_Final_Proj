# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from utils import vision_transformer as vit
from utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler

class JEPA_Model(nn.Module):
    def __init__(
        self,
        device,
        model_name='vit_tiny',
        patch_size=5,
        img_size=65,
        pred_depth=6,
        pred_emb_dim=384,
        action_dim=2,
    ):
        super(JEPA_Model, self).__init__()

        self.observation_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size,
            in_chans=2
        ).to(device)

        embed_dim = self.observation_encoder.embed_dim
        num_heads = self.observation_encoder.num_heads
        
        self.predictor = vit.__dict__["vit_predictor"](
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=pred_depth,
            action_dim = action_dim
        ).to(device)

        self.target_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size,
            in_chans=2
        ).to(device)

        self.target_encoder.load_state_dict(self.observation_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        self.repr_dim = int((img_size // patch_size) ** 2 * embed_dim)
        self.device = device

    def forward(self, states, actions):
        if self.training:
            B, T, C, H, W = states.shape  # B=batch size, T=timesteps, C=channels, H=height, W=width
            flattened_states = states.reshape(B * T, C, H, W)
            states_emb = self.observation_encoder(flattened_states)  # Shape: [B*T, N, D]
            _, N, D = states_emb.shape
            states_emb = states_emb.reshape(B, T, -1, D) # Reshape back to [B, T, N, D]

            states_emb_hist = states_emb[:, :-1, :, :] # Historical embeddings [B, T-1, N, D]
            states_emb_pred_tgt = states_emb[:, 1:, :, :] # Target embeddings [B, T-1, N, D]

            states_emb_pred_tgt = states_emb_pred_tgt.reshape(B * (T-1), N, D) # [B*(T-1), N, D]
            flattened_emb_hist = states_emb_hist.reshape(B * (T-1), N, D) # [B*(T-1), N, D]
            flattened_actions = actions.reshape(B * (T-1), actions.shape[-1]) # [B*(T-1), A]
            states_emb_pred = self.predictor(flattened_emb_hist, flattened_actions)  # Shape: [B*(T-1), N, D] / [B*(T-1), 2]

            loss = nn.MSELoss()(states_emb_pred_tgt, states_emb_pred[:, :-1, :])

            return loss
        else:
            B, T, C, H, W = states.shape # Expecting input [B, 1, C, H, W]
            flattened_states = states.reshape(B * T , C, H, W)  # Flatten to [B, C, H, W]
            states_emb = self.observation_encoder(flattened_states)  # Encode initial state [B, N, D] eg. [64, 169, 192]
            states_emb = states_emb.reshape(B, 1, *states_emb.shape[1:])
            predictions = [states_emb]
            for t in range(actions.shape[1]):
                action_t = actions[:, t, :]
                next_state_emb = self.predictor(states_emb.squeeze(1), action_t)
                next_state_emb = next_state_emb[:, :-1, :].unsqueeze(1)
                predictions.append(next_state_emb)
            pred_encs = torch.cat(predictions, dim=1)
            B, T, N, D = pred_encs.shape
            pred_encs = pred_encs.reshape(B, T, N * D)
            return pred_encs
