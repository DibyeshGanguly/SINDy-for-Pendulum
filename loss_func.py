"""
loss_func.py  –  loss functions for the SINDy physics model
============================================================

Key change vs. original
-----------------------
The original paper used a fixed ODE structure, so no sparsity penalty was
needed.  With the SINDy dictionary approach the model has many more degrees of
freedom, and we need to drive most coefficients to zero.

We therefore add an **L1 regularisation** term on the SINDy coefficient vector:

    L_total = L_MSE(z_encoder, z_phys)  +  λ_KL * L_KL(z_encoder)
                                         +  λ_L1 * ||c||_1

where  c  are the SINDy coefficients in  model.pModel.coeffs .

λ_L1 is controlled via config.yaml (key: ``sindy_l1``).
If that key is absent the default is 1e-3.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kl_divergence(z: torch.Tensor) -> torch.Tensor:
    """KL( q(z) || N(0,1) )  –  prevents encoder collapse."""
    mu     = z.mean(0)
    logvar = torch.log(z.var(0) + 1e-8)          # +eps for numerical safety
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def _sindy_l1(model) -> torch.Tensor:
    """
    L1 norm of the SINDy coefficient vector.

    Works for any model that exposes  .pModel.coeffs  (SINDyModel).
    Returns zero if the physics model has no such attribute.
    """
    if hasattr(model, 'pModel') and hasattr(model.pModel, 'coeffs'):
        return model.pModel.coeffs.abs().sum()
    return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(input_img, outputs, expected_pred, model=None):
    """Plain MSE between encoder trajectory and physics-predicted trajectory."""
    z2_encoder, z2_phys, zroll = outputs
    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[-1])
    z2_phys    = z2_phys.reshape(-1, z2_phys.shape[-1])

    loss = nn.MSELoss()(z2_encoder, z2_phys)
    return loss


def latent_loss(input_img, outputs, expected_pred, model=None):
    """
    Primary training loss:

        L = MSE(z_enc, z_phys)  +  KL(z_enc)  +  λ_L1 * ||c||_1

    λ_L1 is read from config.yaml (key ``sindy_l1``, default 1e-3).
    """
    z2_encoder, z2_phys, zroll = outputs
    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[-1])
    z2_phys    = z2_phys.reshape(-1, z2_phys.shape[-1])

    # --- reconstruction term ---
    mse = nn.MSELoss()(z2_encoder, z2_phys)

    # --- KL regularisation (prevents encoder collapse) ---
    kl = _kl_divergence(z2_encoder)

    # --- SINDy L1 sparsity ---
    try:
        cfg    = OmegaConf.load("config.yaml")
        lam_l1 = float(getattr(cfg, 'sindy_l1', 1e-3))
    except Exception:
        lam_l1 = 1e-3

    l1 = _sindy_l1(model) if model is not None else torch.tensor(0.0)

    return mse + kl + lam_l1 * l1


def latent_loss_multiple(input_img, outputs, expected_pred, model=None):
    """
    Multi-object variant (e.g. two-object spring).

    L = 2 * MSE(z_phys, z_enc)  +  KL(z_enc)  +  λ_L1 * ||c||_1
    """
    d = 2
    z2_encoder, z2_phys, z_renorm = outputs

    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[2])
    z2_phys    = z2_phys.reshape(-1, z2_phys.shape[2])

    mse = nn.MSELoss()(z2_phys, z2_encoder)
    kl  = _kl_divergence(z2_encoder)

    try:
        cfg    = OmegaConf.load("config.yaml")
        lam_l1 = float(getattr(cfg, 'sindy_l1', 1e-3))
    except Exception:
        lam_l1 = 1e-3

    l1 = _sindy_l1(model) if model is not None else torch.tensor(0.0)

    total = d * mse + kl + lam_l1 * l1

    if torch.isnan(mse):
        return kl + lam_l1 * l1
    if torch.isnan(kl):
        return d * mse + lam_l1 * l1

    return total


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def getLoss(loss: str = None):
    """
    Return the appropriate loss function by name.

    The returned callable has signature:
        fn(input_img, outputs, expected_pred, model=None) -> scalar Tensor
    """
    if loss is None:
        try:
            cfg  = OmegaConf.load("config.yaml")
            loss = cfg.loss
        except Exception:
            loss = "latent_loss"

    mapping = {
        "MSE":                  mse_loss,
        "latent_loss":          latent_loss,
        "latent_loss_multiple": latent_loss_multiple,
    }

    if loss not in mapping:
        raise ValueError(
            f"Unknown loss '{loss}'.  Choose from: {list(mapping.keys())}"
        )
    return mapping[loss]


if __name__ == "__main__":
    fn = getLoss()
    print("Default loss function:", fn.__name__)
