# SINDy Extension – Drop-in Files

## What's in this folder

| File | Replaces / adds | Description |
|---|---|---|
| `PhysModels.py` | `src/models/PhysModels.py` | SINDy dictionary physics block |
| `loss_func.py` | `src/loss_func.py` | Adds L1 sparsity penalty |
| `train.py` | `src/train.py` | Passes `model=` into loss; prints discovered eq |
| `generate_synthetic_data.py` | *(new)* | Generates synthetic video datasets |
| `run_sindy.py` | *(new)* | End-to-end demo script |

## Installation

```
cp PhysModels.py         path/to/repo/src/models/PhysModels.py
cp loss_func.py          path/to/repo/src/loss_func.py
cp train.py              path/to/repo/src/train.py
cp generate_synthetic_data.py  path/to/repo/
cp run_sindy.py          path/to/repo/
```

Also add `sindy_l1: 1.0e-3` to your `config.yaml`.

## Quick start

```bash
# 1. Generate synthetic pendulum data
python generate_synthetic_data.py --system pendulum --n_videos 100

# 2. Train the SINDy model (2nd-order for pendulum)
python run_sindy.py --data data/pendulum/video.npy --order 2 --epochs 500

# 3. For LED decay (1st-order)
python generate_synthetic_data.py --system led_decay --n_videos 100
python run_sindy.py --data data/led_decay/video.npy --order 1 --epochs 500
```

## Real video data sources

| Source | URL | Notes |
|---|---|---|
| **Delfys75** (paper's own dataset) | https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction | 75 real videos, 5 systems, includes masks + GT params |
| ODE²VAE pendulum | https://github.com/cagatayyildiz/ODE2VAE | Synthetic pendulum .npy files |
| PhysDreamer | https://huggingface.co/datasets/PhysDreamer | Pendulum + spring |
| YouTube + yt-dlp | any pendulum/spring video | Run through `video2npy.py` + background subtraction |

## How the SINDy block works

The **SINDy (Sparse Identification of Nonlinear Dynamics)** approach replaces
the hard-coded ODE classes (`Damped_oscillation`, `free_fall`, etc.) with a
learned sparse combination of candidate terms.

**Dictionary for 1st-order systems** (`sindy_1st`):

```
du/dt = c0·u + c1·u² + c2·u³ + c3·sin(u) + c4·cos(u) + c5·√|u| + c6·exp(-|u|) + c7·1
```

**Dictionary for 2nd-order systems** (`sindy_2nd`):

```
du/dt = c0·u + c1·u² + c2·u³ + c3·sin(u) + c4·cos(u) + c5·√|u| + c6·1
      + c7·v + c8·v² + c9·u·v          (v = velocity ≈ dz/dt)
```

During training:
- MSE loss forces the physics-integrated trajectory to match the encoder's trajectory
- KL loss prevents encoder collapse
- **L1 loss on coefficients** (`sindy_l1` in config.yaml) drives most terms to zero

After training, call `model.pModel.print_equation()` to see what was discovered.

## Tuning sparsity

`sindy_l1` in `config.yaml` (or `--sindy_l1` CLI flag) controls sparsity:

- **Too small** (e.g. 1e-5): many non-zero terms, equation not sparse
- **Too large** (e.g. 10.0): all terms zeroed out, model can't fit the data
- **Good range**: 1e-4 to 1e-2 typically works

A practical approach: start with 1e-3, check `coeff_evolution.png` to see
which terms survive, then increase until only physically meaningful terms remain.

## Numerical integration

The SINDy block uses explicit **Euler** (1st-order) or **Verlet** (2nd-order)
integration in latent space—the same scheme as the original paper. No PINN is
needed because:

- We are integrating a 1D scalar ODE (the latent state z), not a spatial PDE
- The integration is just a single formula applied at each time step
- The formula is differentiable w.r.t. the coefficients → backprop works

A PINN would be needed if we were solving a *spatial* PDE (e.g. wave equation
on a grid), but since we stay in 1D latent space, Euler/Verlet is sufficient.
