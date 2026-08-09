"""
run_sindy.py
============
End-to-end demo: generate synthetic pendulum data → train SINDy model →
print discovered equation.

Usage
-----
    # Generate data first (skip if you already have data/)
    python generate_synthetic_data.py --system pendulum --n_videos 100

    # Then run this script
    python run_sindy.py --data data/pendulum/video.npy --order 2

    # For a 1st-order system (LED decay)
    python generate_synthetic_data.py --system led_decay --n_videos 100
    python run_sindy.py --data data/led_decay/video.npy --order 1

Notes
-----
- Make sure config.yaml exists (copy the original from the repo).
  At minimum it needs:  train.epochs: 500  and  loss: latent_loss
  You can add  sindy_l1: 1e-3  to control the sparsity penalty.
- The script saves the best model to  ./Results/<experiment>/best_model.pt
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.model import EndPhys
from src.models.PhysModels import SINDyModel
from src import loader, train as train_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True,
                        help='Path to .npy file (N, n_frames, 1, H, W)')
    parser.add_argument('--order',  type=int, default=2, choices=[1, 2],
                        help='ODE order: 1 (LED/Torricelli) or 2 (pendulum/osc)')
    parser.add_argument('--dt',     type=float, default=0.05,
                        help='Time step dt (must match data generation)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr_phys', type=float, default=0.01)
    parser.add_argument('--sindy_l1', type=float, default=1e-3,
                        help='L1 sparsity weight for SINDy coefficients')
    parser.add_argument('--name',   default='sindy_run',
                        help='Experiment name (used for results folder)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Patch config.yaml with CLI args (avoids editing the file manually)
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config.yaml")
    cfg.train.epochs = args.epochs
    cfg.loss         = "latent_loss"
    cfg.sindy_l1     = args.sindy_l1
    OmegaConf.save(cfg, "config.yaml")     # write back temporarily

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[run_sindy] Loading {args.data} …")
    data = np.load(args.data, allow_pickle=True)
    print(f"[run_sindy] Data shape: {data.shape}")
    # shape expected: (N_samples, N_frames, 1, H, W)

    train_dl, val_dl, _, _ = loader.getLoader_folder(data, split=True,
                                                      batch_size=64)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    pmodel_name = f"sindy_{args.order}{'st' if args.order == 1 else 'nd'}"
    model = EndPhys(
        dt       = args.dt,
        pmodel   = pmodel_name,   # "sindy_1st" or "sindy_2nd"
        init_phys= None,          # SINDy doesn't use init_phys
        initw    = True,
    )
    print(f"[run_sindy] Model physics block: {pmodel_name}")
    print(f"[run_sindy] SINDy terms: {model.pModel.term_names}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model, log, params = train_module.train(
        model,
        train_dl,
        val_dl,
        loss_name      = "latent_loss",
        lr_phys        = args.lr_phys,
        experiment_name= args.name,
    )

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Discovered equation (coefficients above threshold 0.01):")
    model.pModel.print_equation(threshold=0.01)
    print("="*60)

    coeffs = model.pModel.coeffs.detach().cpu().numpy()
    names  = model.pModel.term_names
    print("\nFull coefficient table:")
    for n, c in zip(names, coeffs):
        print(f"  {n:12s}  {c:+.6f}")

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    result_dir = f'./Results/{args.name}'
    train_losses = [d['train_loss']      for d in log]
    val_losses   = [d['validation_loss'] for d in log]

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses,   label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'SINDy training – {args.name}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'), dpi=150)
    print(f"\n[run_sindy] Loss curve → {result_dir}/loss_curve.png")

    # SINDy coefficient evolution over training
    if 'sindy_coeffs' in log[0]:
        coeff_history = np.array([d['sindy_coeffs'] for d in log])
        plt.figure(figsize=(12, 4))
        for k, name in enumerate(names):
            plt.plot(coeff_history[:, k], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Coefficient value')
        plt.legend(fontsize=7, ncol=3)
        plt.title('SINDy coefficient evolution')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'coeff_evolution.png'), dpi=150)
        print(f"[run_sindy] Coeff evolution → {result_dir}/coeff_evolution.png")


if __name__ == '__main__':
    main()
