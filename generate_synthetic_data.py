"""
generate_synthetic_data.py
==========================
Generates synthetic video data for testing the SINDy physics model.

Three systems are available:
  - pendulum        (2nd-order, nonlinear: sin(θ))
  - damped_osc      (2nd-order, linear + damping)
  - led_decay       (1st-order, exponential: dz/dt = -α z)

Usage
-----
    python generate_synthetic_data.py --system pendulum --n_videos 200
    python generate_synthetic_data.py --system damped_osc
    python generate_synthetic_data.py --system led_decay

Outputs
-------
  data/<system>/video.npy       shape (N_samples, N_frames, 1, H, W)

The .npy format is identical to what video2npy.py produces, so you can
drop it straight into the existing training pipeline.

Where to find *real* videos
---------------------------
  1. Delfys75 dataset (official paper data):
       https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction
       - 75 real videos: pendulum, torricelli, sliding_block, LED decay,
         free-fall scale.  Includes per-frame masks + ground-truth params.
       - Download and point video2npy.py at the folder.

  2. Other public sources:
       - PhysDreamer / PhysicsBench (pendulum + spring videos)
           https://huggingface.co/datasets/PhysDreamer
       - ODE^2VAE pendulum videos (synthetic, available on GitHub)
           https://github.com/cagatayyildiz/ODE2VAE
       - MIT Moments-in-Time (general motion – needs filtering)
           http://moments.csail.mit.edu
       - YouTube with yt-dlp + OpenCV background subtraction for masking
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical systems
# ---------------------------------------------------------------------------

def _simulate_pendulum(n_frames, dt, alpha=9.81, beta=0.3, theta0=1.0):
    """
    Damped pendulum:  θ'' + β θ' + (α/L) sin(θ) = 0,  with L=1.

    Returns array of angles (radians), length n_frames.
    """
    def ode(t, y):
        theta, omega = y
        dtheta = omega
        domega = -beta * omega - alpha * np.sin(theta)
        return [dtheta, domega]

    t_span = (0, n_frames * dt)
    t_eval = np.linspace(0, n_frames * dt, n_frames)
    sol    = solve_ivp(ode, t_span, [theta0, 0.0], t_eval=t_eval,
                       method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[0]           # angles


def _simulate_damped_osc(n_frames, dt, alpha=4.0, beta=0.5, x0=1.0):
    """
    Damped harmonic oscillator:  x'' + β x' + α x = 0
    """
    def ode(t, y):
        x, v = y
        return [v, -beta * v - alpha * x]

    t_span = (0, n_frames * dt)
    t_eval = np.linspace(0, n_frames * dt, n_frames)
    sol    = solve_ivp(ode, t_span, [x0, 0.0], t_eval=t_eval,
                       method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[0]


def _simulate_led_decay(n_frames, dt, alpha=1.5, z0=1.0):
    """
    LED / exponential decay:  dz/dt = -α z
    Analytic: z(t) = z0 * exp(-α t)
    """
    t = np.linspace(0, n_frames * dt, n_frames)
    return z0 * np.exp(-alpha * t)


# ---------------------------------------------------------------------------
# Render state → greyscale frame
# ---------------------------------------------------------------------------

def _render_pendulum_frame(theta, H=56, W=100, pivot=(50, 10), L=30):
    """
    Draw a simple pendulum bob on a white background.

    theta : angle from vertical (radians)
    Returns uint8 greyscale array (H, W).
    """
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    px, py = pivot
    bx = px + L * np.sin(theta)
    by = py + L * np.cos(theta)

    # string
    ax.plot([px, bx], [H - py, H - by], 'k-', linewidth=1)
    # bob
    circle = plt.Circle((bx, H - by), radius=4, color='black')
    ax.add_patch(circle)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(H, W, 3)
    grey = (0.2989 * buf[:, :, 0]
            + 0.5870 * buf[:, :, 1]
            + 0.1140 * buf[:, :, 2])
    plt.close(fig)
    return (grey / 255.0).astype(np.float32)


def _render_ball_frame(x_norm, H=56, W=100):
    """
    Draw a simple dot whose horizontal (or vertical) position tracks x_norm ∈ [-1, 1].
    Used for oscillator and LED (brightness instead of position).
    """
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Map x_norm to pixel x
    px = W / 2 + x_norm * (W / 2 - 8)
    py = H / 2

    circle = plt.Circle((px, py), radius=5, color='black')
    ax.add_patch(circle)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(H, W, 3)
    grey = (0.2989 * buf[:, :, 0]
            + 0.5870 * buf[:, :, 1]
            + 0.1140 * buf[:, :, 2])
    plt.close(fig)
    return (grey / 255.0).astype(np.float32)


def _render_led_frame(brightness, H=56, W=100):
    """Uniform greyscale frame whose mean brightness tracks the LED decay."""
    val = float(np.clip(brightness, 0.0, 1.0))
    return np.full((H, W), 1.0 - val, dtype=np.float32)   # dark = high decay


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(system, n_videos, n_frames_per_sample, dt, H, W,
                  step=6, **phys_kwargs):
    """
    Simulate many trajectories with slightly varying initial conditions
    and render them into overlapping windows of *n_frames_per_sample* frames.

    Returns
    -------
    np.ndarray, shape (N_samples, n_frames_per_sample, 1, H, W)
    """
    all_samples = []
    rng = np.random.default_rng(42)

    total_frames_needed = n_frames_per_sample * step + 10

    for vid_idx in range(n_videos):
        # Slight variation in initial condition
        ic_scale = rng.uniform(0.5, 1.5)

        if system == 'pendulum':
            theta0 = phys_kwargs.get('theta0', 1.0) * ic_scale
            states = _simulate_pendulum(total_frames_needed, dt,
                                        alpha=phys_kwargs.get('alpha', 9.81),
                                        beta=phys_kwargs.get('beta', 0.3),
                                        theta0=theta0)
            frames_raw = []
            for s in states:
                frames_raw.append(_render_pendulum_frame(s, H=H, W=W))

        elif system == 'damped_osc':
            x0 = phys_kwargs.get('x0', 1.0) * ic_scale
            states = _simulate_damped_osc(total_frames_needed, dt,
                                          alpha=phys_kwargs.get('alpha', 4.0),
                                          beta=phys_kwargs.get('beta', 0.5),
                                          x0=x0)
            # Normalise to [-1, 1] for rendering
            states_norm = states / (np.max(np.abs(states)) + 1e-6)
            frames_raw = [_render_ball_frame(s, H=H, W=W) for s in states_norm]

        elif system == 'led_decay':
            z0 = phys_kwargs.get('z0', 1.0) * ic_scale
            states = _simulate_led_decay(total_frames_needed, dt,
                                         alpha=phys_kwargs.get('alpha', 1.5),
                                         z0=z0)
            frames_raw = [_render_led_frame(s, H=H, W=W) for s in states]

        else:
            raise ValueError(f"Unknown system: {system}")

        # Slice overlapping windows
        frames_np = np.array(frames_raw)            # (T, H, W)
        T = frames_np.shape[0]
        max_start = T - n_frames_per_sample * step
        for start in range(max_start):
            indices = np.arange(start, start + n_frames_per_sample * step, step)
            window  = frames_np[indices]             # (n_frames, H, W)
            window  = window[:, np.newaxis, :, :]   # (n_frames, 1, H, W)
            all_samples.append(window)

        if (vid_idx + 1) % 10 == 0:
            print(f"  Rendered video {vid_idx + 1}/{n_videos} …")

    dataset = np.array(all_samples, dtype=np.float32)  # (N, n_frames, 1, H, W)
    print(f"Dataset shape: {dataset.shape}")
    return dataset


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic physics video data for SINDy training."
    )
    parser.add_argument('--system',    default='pendulum',
                        choices=['pendulum', 'damped_osc', 'led_decay'])
    parser.add_argument('--n_videos',  type=int, default=100,
                        help='Number of independent trajectories to simulate')
    parser.add_argument('--n_frames',  type=int, default=10,
                        help='Frames per sample (window size)')
    parser.add_argument('--dt',        type=float, default=0.05,
                        help='Simulation time step (seconds)')
    parser.add_argument('--height',    type=int, default=56)
    parser.add_argument('--width',     type=int, default=100)
    parser.add_argument('--step',      type=int, default=6,
                        help='Frame stride (skip frames to increase dt_eff)')
    parser.add_argument('--out_dir',   default='data',
                        help='Output directory (will create <out_dir>/<system>/)')
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, args.system)
    os.makedirs(out_path, exist_ok=True)

    print(f"[generate] System : {args.system}")
    print(f"[generate] Videos : {args.n_videos}")
    print(f"[generate] dt     : {args.dt}")

    dataset = build_dataset(
        system=args.system,
        n_videos=args.n_videos,
        n_frames_per_sample=args.n_frames,
        dt=args.dt,
        H=args.height,
        W=args.width,
        step=args.step,
    )

    save_path = os.path.join(out_path, 'video.npy')
    np.save(save_path, dataset)
    print(f"[generate] Saved → {save_path}   shape={dataset.shape}")

    # Save a preview strip
    sample = dataset[0]                       # (n_frames, 1, H, W)
    strip  = np.hstack([sample[i, 0] for i in range(sample.shape[0])])
    preview_path = os.path.join(out_path, 'preview.png')
    plt.figure(figsize=(20, 3))
    plt.imshow(strip, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'{args.system} – first sample (frames left→right)')
    plt.tight_layout()
    plt.savefig(preview_path, dpi=100)
    plt.close()
    print(f"[generate] Preview → {preview_path}")


if __name__ == '__main__':
    main()
