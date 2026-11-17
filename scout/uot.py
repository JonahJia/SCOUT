
from __future__ import annotations

from typing import Optional, Tuple

import torch

def generalized_kl(p: torch.Tensor, q: torch.Tensor, delta: float = 1e-8) -> torch.Tensor:
    """
    Generalized KL divergence used in SCOUT (Eq. (7)):
      KL(p||q) = sum_t [ p_t log(p_t/q_t) - p_t + q_t ].

    p, q must be non-negative. We add a small delta for stability.
    """
    p_ = p.clamp_min(delta)
    q_ = q.clamp_min(delta)
    return (p_ * (p_.log() - q_.log()) - p_ + q_).sum()

def uot_sinkhorn_log(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    rho1: float,
    rho2: float,
    n_iters: int = 20,
    delta: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Log-domain generalized Sinkhorn for Unbalanced OT (SCOUT Algorithm 1, Eq. (12)-(14)).

    Args:
      C: cost matrix, shape (M, Mp)
      a: source weights, shape (M,) (sum to 1)
      b: target weights, shape (Mp,) (sum to 1)
      eps: entropic regularization epsilon
      rho1, rho2: KL penalties for marginals
      n_iters: iterations L
      delta: floor constant

    Returns:
      T: coupling matrix (M, Mp)
      logT: log coupling (M, Mp)
    """
    assert C.dim() == 2, "C must be (M, Mp)"
    M, Mp = C.shape
    device = C.device
    dtype = C.dtype

    logK = -C / eps
    tau1 = rho1 / (rho1 + eps)
    tau2 = rho2 / (rho2 + eps)

    f = torch.zeros((M,), device=device, dtype=dtype)  # log u
    g = torch.zeros((Mp,), device=device, dtype=dtype)  # log v

    loga = (a + delta).log()
    logb = (b + delta).log()

    for _ in range(n_iters):
        # logKv_i = LSE_j(logK_ij + g_j)
        logKv = torch.logsumexp(logK + g.unsqueeze(0), dim=1)
        f = tau1 * (loga - logKv)

        # logKTu_j = LSE_i(logK_ij + f_i)
        logKTu = torch.logsumexp(logK + f.unsqueeze(1), dim=0)
        g = tau2 * (logb - logKTu)

        # Re-centering stabilization
        c = f.mean()
        f = f - c
        g = g + c

    logT = f.unsqueeze(1) + logK + g.unsqueeze(0)
    T = torch.exp(logT)
    return T, logT

def uot_objective(
    T: torch.Tensor,
    logT: torch.Tensor,
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    rho1: float,
    rho2: float,
    delta: float = 1e-8,
) -> torch.Tensor:
    """
    Evaluate the full UOT objective Eq. (6) at a given coupling T* (SCOUT Sec. 3.3).

    Args:
      T, logT: coupling and its log
      C: cost matrix
      a, b: weights
      eps, rho1, rho2: hyperparams

    Returns:
      scalar objective value
    """
    # Matching cost
    match = (T * C).sum()

    # Entropic regularization: eps * sum_ij T_ij (log T_ij - 1)
    ent = eps * (T * (logT - 1.0)).sum()

    r = T.sum(dim=1)  # row marginals
    c = T.sum(dim=0)  # col marginals

    kl_r = generalized_kl(r, a, delta=delta)
    kl_c = generalized_kl(c, b, delta=delta)

    return match + ent + rho1 * kl_r + rho2 * kl_c
