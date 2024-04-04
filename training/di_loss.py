# Copyright (c) 2023, Weijian Luo, Peking University <pkulwj1994@icloud.com>. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train one-step diffusion-based generative model using the techniques described in the
paper "Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models"
https://github.com/pkulwj1994/diff_instruct

Code was modified from paper ""Elucidating the Design Space of Diffusion-Based Generative Models""
https://github.com/NVlabs/edm
"""

"""Loss functions used in the paper
"Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models"."""

import torch
from torch_utils import persistence
from torch.distributions.log_normal import LogNormal
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Diff-Instruct: A Universal Approach for Transferring Knowledge of Diffusion Models".

@persistence.persistent_class
class DI_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
    def gloss(self, Sd, Sg, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = 1.0
        
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, torch.zeros(images.shape[0], 9).to(images.device))
        n = torch.randn_like(y) * sigma
        
        Sg.train(), Sd.train()
        with torch.no_grad():
            cuda_rng_state = torch.cuda.get_rng_state()
            Dd_yn = Sd(y + n, sigma, labels, augment_labels=augment_labels)
            torch.cuda.set_rng_state(cuda_rng_state)
            Dg_yn = Sg(y + n, sigma, labels, augment_labels=augment_labels)
        Sd.eval()
        
        loss = weight * ((Dg_yn - Dd_yn) * images)
        
        return loss

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        net.train()
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss