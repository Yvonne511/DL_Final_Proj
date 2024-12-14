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
        action_dim=2
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

    def forward(self, s_t, o_t1, action):
        s_t_pred = self.predictor(s_t, action)
        s_t_target = self.target_encoder(o_t1)

        l = self.loss(s_t_pred, s_t_target)
        
        return s_t_pred, l
