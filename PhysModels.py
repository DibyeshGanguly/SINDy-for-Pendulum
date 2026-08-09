"""
PhysModels.py  –  SINDy-style sparse physics block
====================================================
Replaces the hard-coded ODE classes (Damped_oscillation, free_fall, …) with a
single *SINDyModel* that learns which terms from a user-supplied dictionary
actually appear in the governing equation.

Architecture recap
------------------
Given a latent trajectory  z_0, z_1, …, z_N  produced by the CNN encoder,
the physics block predicts the *next* latent state from the previous one(s):

    z_{t+1} = z_t  +  dt * Σ_k  c_k * φ_k(z_t)          (Euler, 1st-order)

    z_{t+2} = z_{t+1} + (z_{t+1} - z_t)
              + dt * Σ_k  c_k * φ_k(z_{t+1})              (Verlet, 2nd-order)

The coefficients c_k are learnable parameters; L1 regularisation (added to the
loss in loss_func.py) drives most of them to zero, recovering a sparse PDE.

Dictionary terms (1-D latent state  u = z_t)
--------------------------------------------
    u          – state itself               (identity)
    u²         – quadratic nonlinearity
    u³         – cubic nonlinearity
    sin(u)     – trigonometric
    cos(u)     – trigonometric
    sqrt|u|    – square-root (e.g. Torricelli)
    exp(-u)    – exponential decay
    1          – constant forcing / bias

For 2nd-order systems the velocity  v ≈ (z_{t+1} - z_t)/dt  is also exposed:
    v, v², u·v

Usage
-----
    from src.models.PhysModels import SINDyModel, getModel

    # 1st-order (LED decay, Torricelli, …)
    model = SINDyModel(order=1)

    # 2nd-order (pendulum, damped oscillator, …)
    model = SINDyModel(order=2)

    # or use the legacy factory wrapper
    model = getModel("sindy_1st")
    model = getModel("sindy_2nd")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dictionary of basis functions
# ---------------------------------------------------------------------------

def _build_library_1st(u: torch.Tensor) -> torch.Tensor:
    """
    Build the SINDy library for a 1st-order system.

    Parameters
    ----------
    u : Tensor, shape (batch, 1)
        Current latent state.

    Returns
    -------
    Tensor, shape (batch, n_terms)
        Each column is one candidate basis function evaluated at u.

    Terms
    -----
    idx  name        expression
     0   u           u
     1   u^2         u²
     2   u^3         u³
     3   sin(u)      sin(u)
     4   cos(u)      cos(u)
     5   sqrt|u|     sqrt(|u| + eps)
     6   exp(-u)     exp(-|u|)      (use |u| to avoid blow-up)
     7   1           constant
    """
    eps = 1e-6
    u2   = u * u
    u3   = u2 * u
    sinu = torch.sin(u)
    cosu = torch.cos(u)
    sqrtu = torch.sqrt(torch.abs(u) + eps)
    expu  = torch.exp(-torch.abs(u))
    ones  = torch.ones_like(u)

    # shape: (batch, 8)
    return torch.cat([u, u2, u3, sinu, cosu, sqrtu, expu, ones], dim=-1)


def _build_library_2nd(u: torch.Tensor,
                        v: torch.Tensor) -> torch.Tensor:
    """
    Build the SINDy library for a 2nd-order system.

    Parameters
    ----------
    u : Tensor, shape (batch, 1)   current position
    v : Tensor, shape (batch, 1)   current velocity ≈ (u_t - u_{t-1})/dt

    Returns
    -------
    Tensor, shape (batch, n_terms)

    Terms (position-based + velocity-based)
    ----------------------------------------
    idx  name         expression
     0   u            u
     1   u^2          u²
     2   u^3          u³
     3   sin(u)       sin(u)
     4   cos(u)       cos(u)
     5   sqrt|u|      sqrt(|u|)
     6   1            constant
     7   v            v  (velocity)
     8   v^2          v²
     9   u*v          u·v  (coupling)
    """
    eps = 1e-6
    u2   = u * u
    u3   = u2 * u
    sinu = torch.sin(u)
    cosu = torch.cos(u)
    sqrtu = torch.sqrt(torch.abs(u) + eps)
    ones  = torch.ones_like(u)
    v2    = v * v
    uv    = u * v

    # shape: (batch, 10)
    return torch.cat([u, u2, u3, sinu, cosu, sqrtu, ones, v, v2, uv], dim=-1)


# ---------------------------------------------------------------------------
# SINDy physics module
# ---------------------------------------------------------------------------

class SINDyModel(nn.Module):
    """
    SINDy-style sparse physics model.

    Learns a sparse linear combination of basis functions to predict the
    evolution of the latent state.  L1 regularisation on the coefficients
    (applied externally in the loss) promotes sparsity.

    Parameters
    ----------
    order : {1, 2}
        Order of the system.
        1 → first-order ODE  (e.g. LED decay, Torricelli flow)
        2 → second-order ODE (e.g. pendulum, damped oscillator)
    init_scale : float
        Initial magnitude of the coefficient vector.  Smaller values make
        sparsification easier at the start of training.
    """

    # Number of library terms per order
    _N_TERMS = {1: 8, 2: 10}

    def __init__(self, order: int = 1, init_scale: float = 0.1):
        super().__init__()

        assert order in (1, 2), "order must be 1 or 2"
        self.order = order
        n_terms = self._N_TERMS[order]

        # Learnable coefficients  c_k  for each basis function
        # Initialise near zero so sparsity regularisation can prune easily
        self.coeffs = nn.Parameter(
            torch.randn(n_terms) * init_scale
        )

    # ------------------------------------------------------------------
    # Library names (for printing / inspection)
    # ------------------------------------------------------------------

    @property
    def term_names(self):
        if self.order == 1:
            return ["u", "u²", "u³", "sin(u)", "cos(u)", "√|u|", "exp(-|u|)", "1"]
        else:
            return ["u", "u²", "u³", "sin(u)", "cos(u)", "√|u|", "1",
                    "v", "v²", "u·v"]

    def print_equation(self, threshold: float = 1e-2):
        """Print the discovered equation (terms above *threshold*)."""
        c = self.coeffs.detach().cpu()
        names = self.term_names
        active = [(names[i], c[i].item()) for i in range(len(c))
                  if abs(c[i].item()) > threshold]
        if not active:
            print("du/dt = 0  (all coefficients below threshold)")
            return
        terms = " + ".join(f"({val:.4f})·{name}" for name, val in active)
        print(f"du/dt = {terms}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Predict next latent state via explicit numerical integration.

        Parameters
        ----------
        z   : Tensor
            For order=1: shape (batch, 1, latent_dim)   – current state only
            For order=2: shape (batch, 2, latent_dim)   – [z_{t-1}, z_t]
        dt  : float
            Time step.

        Returns
        -------
        Tensor, shape (batch, 1, latent_dim)
            Predicted next state  z_{t+1}.
        """
        device = z.device
        dt_t = torch.tensor([dt], dtype=torch.float32, device=device)

        if self.order == 1:
            # z shape: (batch, 1, latent_dim)
            u = z  # (batch, 1, latent_dim)

            # Flatten latent dim for library evaluation (assume latent_dim=1)
            u_flat = u.reshape(u.shape[0], -1)          # (batch, latent_dim)

            lib  = _build_library_1st(u_flat)            # (batch, n_terms)
            dudt = lib @ self.coeffs                     # (batch,)

            # Euler step
            z_next = u_flat + dt_t * dudt.unsqueeze(-1) # (batch, latent_dim)
            return z_next.unsqueeze(1)                   # (batch, 1, latent_dim)

        else:  # order == 2
            # z shape: (batch, 2, latent_dim)
            u_prev = z[:, 0:1, :]   # z_{t-1}
            u_curr = z[:, 1:2, :]   # z_t

            u_flat    = u_curr.reshape(u_curr.shape[0], -1)   # (batch, ld)
            u_prev_flat = u_prev.reshape(u_prev.shape[0], -1) # (batch, ld)

            # Approximate velocity
            v_flat = (u_flat - u_prev_flat) / (dt_t + 1e-8)   # (batch, ld)

            lib  = _build_library_2nd(u_flat, v_flat)          # (batch, n_terms)
            dudt = lib @ self.coeffs                           # (batch,)

            # Verlet-style step (consistent with original paper)
            z_next = (u_flat
                      + (u_flat - u_prev_flat)
                      + dt_t * dt_t * dudt.unsqueeze(-1))      # (batch, ld)

            return z_next.unsqueeze(1)                         # (batch, 1, ld)


# ---------------------------------------------------------------------------
# Convenience factory (keeps backward-compat with existing main.py)
# ---------------------------------------------------------------------------

def getModel(name: str, init_phys=None) -> nn.Module:
    """
    Factory function.  Accepts legacy names (for backward compatibility)
    and new SINDy names.

    New names
    ---------
    "sindy_1st"  →  SINDyModel(order=1)
    "sindy_2nd"  →  SINDyModel(order=2)

    Legacy names still work so existing experiments don't break:
    "Damped_oscillation", "pendulum", "free_fall", "led", "torricelli",
    "sliding_block"   → all mapped to the appropriate SINDy order.
    """
    # --- new explicit names ---
    if name == "sindy_1st":
        return SINDyModel(order=1)
    if name == "sindy_2nd":
        return SINDyModel(order=2)

    # --- legacy names (mapped to SINDy for consistency) ---
    FIRST_ORDER  = {"led", "torricelli", "dyn_1storder", "IntegratedFire", "lineal"}
    SECOND_ORDER = {"Damped_oscillation", "Oscillation", "Sprin_ode",
                    "gravity_ode", "double_pendulum", "pendulum",
                    "sliding_block", "bouncing_ball", "dropped_ball", "free_fall"}

    if name in FIRST_ORDER:
        print(f"[PhysModels] Mapping legacy '{name}' → SINDyModel(order=1)")
        return SINDyModel(order=1)
    if name in SECOND_ORDER:
        print(f"[PhysModels] Mapping legacy '{name}' → SINDyModel(order=2)")
        return SINDyModel(order=2)

    raise ValueError(
        f"Unknown model name: '{name}'.  "
        f"Use 'sindy_1st' or 'sindy_2nd', or a legacy name."
    )
