from __future__ import annotations

import torch


def nhp_loglikelihood_mc(
    *,
    model,
    times: torch.Tensor,  # (N,)
    types: torch.Tensor,  # (N,)
    marks: torch.Tensor,  # (N,)
    t_start: float,
    t_end: float,
    mc_samples_per_interval: int,
) -> torch.Tensor:
    """Monte Carlo approximation of log-likelihood for multivariate point process.

    L = sum_i log λ_{k_i}(t_i) - ∫_{t_start}^{t_end} Σ_k λ_k(t) dt

    We estimate integral by sampling uniformly within each inter-event interval.
    """
    device = times.device
    eps = torch.tensor(1e-8, device=device)

    # 1) log intensity at event times
    # Query intensities at each event time (right before the event).
    lam_at_events = model.forward_intensity_path(times, types, marks, query_times=times)
    idx = torch.arange(types.numel(), device=device)
    lam_k = lam_at_events[idx, types]
    log_term = torch.log(lam_k + eps).sum()

    # 2) integral term
    # Build intervals: [t0, t1), [t1, t2), ... plus [tN, t_end]
    t0 = torch.tensor(float(t_start), device=device)
    t_end_t = torch.tensor(float(t_end), device=device)
    all_ts = torch.cat([t0[None], times, t_end_t[None]], dim=0)
    integral = torch.tensor(0.0, device=device)

    for a, b in zip(all_ts[:-1], all_ts[1:]):
        dt = b - a
        if dt <= 0:
            continue
        # Sample points in (a, b)
        u = torch.rand(mc_samples_per_interval, device=device)
        tq = a + u * dt
        tq, _ = torch.sort(tq)
        lam = model.forward_intensity_path(times, types, marks, query_times=tq)  # (M, K)
        integral = integral + (lam.sum(dim=-1).mean() * dt)

    return log_term - integral


