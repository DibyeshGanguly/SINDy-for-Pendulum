"""
train.py  –  training loop for the SINDy physics model
=======================================================

Changes vs. original
--------------------
1. The loss function is now called with  loss_fn(..., model=model)  so that
   the SINDy L1 regularisation term can access  model.pModel.coeffs .

2. After training, ``print_discovered_equation()`` prints the sparse equation
   that was discovered.

3. Minor: loss_func signatures now accept a keyword-only  model=  argument
   (backward-compatible because we guard with  model=None  default).
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import os

from src import loss_func


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def freeze(model, type_):
    """Freeze or unfreeze parts of the model."""
    for name, param in model.named_parameters():
        param.requires_grad = True

    if type_ == 'encoder':
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False

    elif type_ == 'encoder-decoder':
        for name, param in model.named_parameters():
            if name in ('pModel.alpha', 'pModel.beta',
                        'pModel.coeffs'):          # <-- SINDy coeffs
                param.requires_grad = False

    elif type_ == 'pModel':
        for name, param in model.named_parameters():
            if name not in ('pModel.k', 'pModel.eq_distance',
                            'pModel.coeffs'):      # <-- SINDy coeffs
                param.requires_grad = False

    return model


def print_discovered_equation(model):
    """Print the SINDy equation if the physics model supports it."""
    if hasattr(model, 'pModel') and hasattr(model.pModel, 'print_equation'):
        print("\n[SINDy] Discovered equation:")
        model.pModel.print_equation(threshold=1e-2)
    elif hasattr(model, 'pModel') and hasattr(model.pModel, 'alpha'):
        alpha = model.pModel.alpha[0].detach().cpu().item()
        print(f"\n[Physics] alpha = {alpha:.6f}")
        if hasattr(model.pModel, 'beta'):
            beta = model.pModel.beta[0].detach().cpu().item()
            print(f"[Physics] beta  = {beta:.6f}")


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, device='cpu',
                print_loss=False):
    model.train()
    total_loss = 0.0

    for batch_idx, (input_data, _) in enumerate(loader):
        x = input_data.to(device=device, dtype=torch.float)

        outputs = model(x)

        # Pass model so loss can access SINDy coefficients for L1 penalty
        loss = loss_fn(x, outputs, None, model=model)

        if torch.isnan(loss):
            if print_loss:
                print(f"  NaN loss at batch {batch_idx}")
            return float('nan')

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_epoch(model, loader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_data, _ in loader:
            x = input_data.to(device=device, dtype=torch.float)
            outputs = model(x)
            loss = loss_fn(x, outputs, None, model=model)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader,
          type_='normal',
          init_phys=1.0,
          loss_name=None,
          lr_phys=0.01,
          experiment_name='experiment'):
    """
    Train the model.

    Parameters
    ----------
    model        : nn.Module   The EndPhys (or similar) model.
    train_loader : DataLoader
    val_loader   : DataLoader
    type_        : str         'normal' or 'dynamic' (alternating freeze).
    init_phys    : float       Initial LR for physics parameters.
    loss_name    : str or None Name of loss function (see loss_func.getLoss).
    lr_phys      : float       Learning rate for the physics / SINDy block.
    experiment_name : str      Used for saving checkpoints.

    Returns
    -------
    model, log, params
    """
    cfg = OmegaConf.load("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    model.to(device)
    num_epochs = cfg.train.epochs
    patience   = getattr(cfg.train, 'patience', 50)
    loss_fn    = loss_func.getLoss(loss_name)

    # ------------------------------------------------------------------
    # Optimiser: separate LR for encoder vs SINDy / physics parameters
    # ------------------------------------------------------------------
    phys_params = []
    enc_params  = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc_params.append(param)
        else:
            phys_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': enc_params,  'lr': 1e-3,    'name': 'encoder'},
        {'params': phys_params, 'lr': lr_phys, 'name': 'physics'},
    ])

    # ------------------------------------------------------------------
    # Result directory
    # ------------------------------------------------------------------
    result_dir = f'./Results/{experiment_name}'
    os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses = []
    val_losses   = []
    log          = []
    early_stop   = 0
    best_val     = float('inf')
    best_model_state = None

    # Initial eval
    train_loss = evaluate_epoch(model, train_loader, loss_fn, device=device)
    val_loss   = evaluate_epoch(model, val_loader,   loss_fn, device=device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Initial  \t train: {train_loss:.6f}  \t val: {val_loss:.6f}")

    dict_log = {"train_loss": train_loss, "validation_loss": val_loss}
    _log_phys_params(model, dict_log)
    log.append(dict_log)

    for epoch in range(1, num_epochs + 1):

        print_loss = (epoch % 100 == 0)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer,
                                 device=device, print_loss=print_loss)
        val_loss   = evaluate_epoch(model, val_loader, loss_fn, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        dict_log = {"train_loss": train_loss, "validation_loss": val_loss}
        _log_phys_params(model, dict_log)
        log.append(dict_log)

        if epoch % 100 == 0:
            print(f"epoch: {epoch} \t train: {train_loss:.6f} "
                  f"\t val: {val_loss:.6f}")

        if np.isnan(train_loss):
            print(f"[train] NaN loss at epoch {epoch}. Restoring best model.")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            return model, log, _extract_params(model)

        # Dynamic freeze schedule (alternating encoder / physics)
        if type_ == 'dynamic':
            if epoch % 2 == 0:
                model = freeze(model, 'encoder')
            if epoch % 2 == 1 or epoch > 3 * num_epochs / 4:
                model = freeze(model, 'encoder-decoder')

        # Track best model
        if val_loss < best_val:
            best_val = val_loss
            best_model_state = {k: v.clone()
                                for k, v in model.state_dict().items()}
            torch.save(model.state_dict(),
                       os.path.join(result_dir, 'best_model.pt'))
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= patience:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print_discovered_equation(model)

    params = _extract_params(model)
    return model, log, params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_phys_params(model, d: dict):
    """Append physics parameters to the log dict."""
    if not hasattr(model, 'pModel'):
        return
    pm = model.pModel
    if hasattr(pm, 'coeffs'):
        # SINDy: log the full coefficient vector as a list
        d['sindy_coeffs'] = pm.coeffs.detach().cpu().tolist()
    else:
        for name, value in pm.named_parameters():
            if 'encoder' not in name:
                try:
                    d[name] = value[0].detach().cpu().numpy().item()
                except Exception:
                    pass


def _extract_params(model):
    """Return a dict of the learned physics parameters."""
    if not hasattr(model, 'pModel'):
        return {}
    pm = model.pModel
    if hasattr(pm, 'coeffs'):
        return {'sindy_coeffs': pm.coeffs.detach().cpu().tolist()}
    out = {}
    for name, value in pm.named_parameters():
        try:
            out[name] = value[0].detach().cpu().item()
        except Exception:
            pass
    return out
