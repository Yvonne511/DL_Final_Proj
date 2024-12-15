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
import itertools

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
        momentum_scheduler=None
    ):
        super(JEPA_Model, self).__init__()

        self.observation_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size
        ).to(device)
        embed_dim = self.observation_encoder.embed_dim
        num_heads = self.observation_encoder.num_heads

        self.predictor = vit.__dict__["vit_predictor"](
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=pred_depth,
            action_dim=action_dim
        ).to(device)

        self.target_encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size
        ).to(device)

        self.target_encoder.load_state_dict(self.observation_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.device = device

        self.loss = nn.MSELoss()
        self.repr_dim = int((img_size // patch_size) ** 2 * embed_dim)
        self.momentum_scheduler = momentum_scheduler or itertools.cycle([0.999])

        self.online_transform = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1))
        ])

        self.target_transform = T.Compose([
            T.RandomResizedCrop(64, scale=(0.8, 1.0)),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05))
        ])

    def momentum_update(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)  # Get current momentum value
            for param_q, param_k in zip(self.observation_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.data)

    def compute_loss(self, obs, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            loss for training and validation
        """
        s_t = self.observation_encoder(obs[:, 0])
        total_loss = 0.0

        for t in range(obs.size(1) - 1):
            action_t = actions[:, t]
            s_t_pred = self.predictor(s_t, action_t)

            o_t1 = obs[:, t+1]
            o_t1 = self.online_transform(o_t1)

            with torch.no_grad():
                s_t_target = self.target_encoder(o_t1).detach()
                s_t_target = self.target_transform(s_t_target)
            
            loss = self.loss(s_t_pred, s_t_target)

            total_loss += loss
            s_t = s_t_pred.detach()
            self.momentum_update()

        return total_loss


    def forward(self, states, actions):
        """
        Args:
            states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        s_t = self.observation_encoder(states[:, 0])
        B, N, D = s_t.shape
        predictions = [s_t.unsqueeze(1)]

        for t in range(actions.size(1)):
            action_t = actions[:, t]
            s_t_pred = self.predictor(s_t, action_t)
            predictions.append(s_t_pred.unsqueeze(1))
            s_t = s_t_pred.detach()

        predictions = torch.cat(predictions, dim=1)
        B, T, N, D = predictions.shape
        predictions = predictions.reshape(B, T, N*D)
        return predictions
