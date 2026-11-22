"""Alternative price models, all with the same Model signature as validate.gbm_model.

Every model uses zero drift (per the drift-uncertainty finding): daily log returns are centred
so their mean is -sigma^2/2, which keeps the expected price equal to today's, like GBM.
"""

from functools import partial

import numpy as np
from scipy import stats

from mc.validate import Model, _to_prices, gbm_model


def _centre(hist: np.ndarray) -> np.ndarray:
    return hist - hist.mean() - 0.5 * hist.var(ddof=1)


def student_t_model(hist: np.ndarray, s0: float, days: int, n_paths: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Student-t daily moves: a bell curve with fatter tails, fitted to the history."""
    sd = float(np.std(hist, ddof=1))
    df = max(float(stats.t.fit(hist)[0]), 3.0)  # below ~3 the variance is barely finite
    scale = sd * np.sqrt((df - 2) / df)  # so the t draws have the history's volatility
    steps = -0.5 * sd**2 + scale * rng.standard_t(df, (n_paths, days))
    return _to_prices(s0, steps)


def jump_diffusion_model(hist: np.ndarray, s0: float, days: int, n_paths: int,
                         rng: np.random.Generator, k: float = 3.0) -> np.ndarray:
    """Merton jump-diffusion: calm normal wiggles plus occasional sudden gaps.

    Days beyond k standard deviations are treated as jumps; their frequency and size are
    estimated from the history, and the wiggle volatility from the remaining days.
    """
    sd = float(np.std(hist, ddof=1))
    dev = hist - hist.mean()
    is_jump = np.abs(dev) > k * sd
    if is_jump.sum() < 2:
        return gbm_model(hist, s0, days, n_paths, rng)
    lam = float(is_jump.mean())
    m_j, s_j = float(dev[is_jump].mean()), float(dev[is_jump].std(ddof=1))
    sd_d = float(dev[~is_jump].std(ddof=1))
    n_jumps = rng.poisson(lam, (n_paths, days))
    jumps = m_j * n_jumps + s_j * np.sqrt(n_jumps) * rng.standard_normal((n_paths, days))
    steps = sd_d * rng.standard_normal((n_paths, days)) + jumps
    steps += -0.5 * sd**2 - lam * m_j  # zero drift, same convention as GBM
    return _to_prices(s0, steps)


def bootstrap_model(hist: np.ndarray, s0: float, days: int, n_paths: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Replay history: each simulated day is a randomly picked real day."""
    c = _centre(np.asarray(hist, dtype=float))
    return _to_prices(s0, c[rng.integers(0, len(c), (n_paths, days))])


def block_bootstrap_model(hist: np.ndarray, s0: float, days: int, n_paths: int,
                          rng: np.random.Generator, block: int = 20) -> np.ndarray:
    """Replay history in runs of `block` consecutive real days, so calm and stormy spells
    stay together (volatility clustering). Wraps around the end of the history."""
    c = _centre(np.asarray(hist, dtype=float))
    n_blocks = -(-days // block)
    starts = rng.integers(0, len(c), (n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :days] % len(c)
    return _to_prices(s0, c[idx])


MODELS: dict[str, Model] = {
    "GBM (normal)": gbm_model,
    "Student-t": student_t_model,
    "Jump-diffusion": jump_diffusion_model,
    "Bootstrap": bootstrap_model,
    "Block bootstrap": partial(block_bootstrap_model, block=20),
}
