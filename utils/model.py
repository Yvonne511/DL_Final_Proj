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
        model_name='vit_custom',
        predictor_model_name='vit_custom_predictor', 
        patch_size=5,
        img_size=65,
        pred_depth=4,
        pred_num_heads=6,
        action_dim=2,
        momentum_scheduler=None,
        embed_dim=128,
        encoder_depth=8,
        encoder_num_heads=6,
        mlp_ratio=6,

    ):
        
        
        super(JEPA_Model, self).__init__()

        self.observation_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            patch_size=patch_size,
            in_chans=2,
            mlp_ratio=mlp_ratio
        ).to(device)

        embed_dim = self.observation_encoder.embed_dim
        num_heads = self.observation_encoder.num_heads
        
        self.predictor = vit.__dict__[predictor_model_name](
            embed_dim=embed_dim,
            num_heads=pred_num_heads,
            depth=pred_depth,
            action_dim = action_dim,
            mlp_ratio=mlp_ratio
        ).to(device)

        self.target_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size,
            in_chans=2,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio
        ).to(device)

        self.target_encoder.load_state_dict(self.observation_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        self.repr_dim = int((img_size // patch_size) ** 2 * embed_dim)
        self.device = device
        self.momentum_scheduler = momentum_scheduler or (x for x in [0.999] * 1000)

    def momentum_update(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)  # Get current momentum value
            for param_q, param_k in zip(self.observation_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.data)

    def variance_loss(self, embeddings, epsilon=1e-4, target_std=0.8):
        # embeddings: [B, D]
        # Compute variance along the batch dimension
        var = embeddings.var(dim=0) + epsilon
        std = torch.sqrt(var)
        # Hinge-like penalty to push std above target_std
        var_loss = torch.mean(nn.functional.relu(target_std - std))
        return var_loss

    def covariance_loss(self, embeddings, epsilon=1e-4):
        # embeddings: [B, D]
        B, D = embeddings.size()
        embeddings = embeddings - embeddings.mean(dim=0)
        cov = (embeddings.T @ embeddings) / (B - 1) + epsilon
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / D
        return cov_loss

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
            states_emb_pred = states_emb_pred[:, :-1, :] 

            f1_loss = nn.functional.smooth_l1_loss(states_emb_pred_tgt, states_emb_pred)

            states_emb_pred_pooled = states_emb_pred.mean(dim=1)
            states_emb_pred_tgt_pooled = states_emb_pred_tgt.mean(dim=1) 

            var_loss_pred = self.variance_loss(states_emb_pred_pooled)
            var_loss_tgt = self.variance_loss(states_emb_pred_tgt_pooled)
            cov_loss_pred = self.covariance_loss(states_emb_pred_pooled)
            cov_loss_tgt = self.covariance_loss(states_emb_pred_tgt_pooled)

            var_loss = var_loss_pred + var_loss_tgt
            cov_loss = cov_loss_pred + cov_loss_tgt
            lambda_f1 = 10
            lambda_var = 1
            lambda_cov = 10
            loss = lambda_f1 * f1_loss + lambda_var * var_loss + lambda_cov * cov_loss

            self.momentum_update()
            return loss, f1_loss, var_loss, cov_loss
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
