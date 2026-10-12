"""Section 4.5 Hierarchical anomaly scoring.

This module provides the two building blocks of hierarchical scoring as pure functions:

1. Gaussian mixture log-density -- every ball B_k = N(mu_k, Sigma_k) at the current
   level is treated as a mixture component, and the anomaly score of x is -log p(x).
2. Ball-distribution entropy -> level weights -- a level whose mass is spread more
   evenly receives a larger weight, after normalization.

The density is evaluated with a log-sum-exp formulation (no sklearn private API);
it is numerically equivalent to sklearn.mixture.GaussianMixture.score_samples.
"""
import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


def regularize_covariance(covariance, eps=1e-8):
    """Keep the covariance numerically positive definite (same rule as inside GranularBall)."""
    covariance = np.asarray(covariance, dtype=float)
    min_eigenval = np.min(np.linalg.eigvalsh(covariance))
    if min_eigenval < eps:
        reg_factor = max(eps - min_eigenval, eps)
        covariance = covariance + np.eye(covariance.shape[0]) * reg_factor
    return covariance


def log_gaussian_density(data, mean, covariance, regularize=True):
    """log N(x; mu, Sigma), shape (n_samples,).

    Computed directly from the definition:
        -0.5 * [ d*log(2*pi) + log|Sigma| + (x-mu)^T Sigma^-1 (x-mu) ]
    """
    data = np.atleast_2d(data)
    n_features = data.shape[1]
    covariance = regularize_covariance(covariance) if regularize else np.asarray(covariance, dtype=float)

    _, log_det = np.linalg.slogdet(covariance)
    diff = data - np.asarray(mean, dtype=float)          # (n, d)
    solved = np.linalg.solve(covariance, diff.T)         # Sigma^-1 (x-mu), shape (d, n)
    maha = np.sum(diff.T * solved, axis=0)               # (n,)
    return -0.5 * (n_features * _LOG_2PI + log_det + maha)


def mixture_log_density(data, means, covariances, weights):
    """log sum_k w_k N(x; mu_k, Sigma_k) via log-sum-exp, shape (n_samples,)."""
    data = np.atleast_2d(data)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    log_weights = np.log(np.maximum(weights, 1e-300))

    components = np.stack(
        [log_gaussian_density(data, mean, cov) for mean, cov in zip(means, covariances)],
        axis=1,
    ) + log_weights[None, :]

    row_max = components.max(axis=1, keepdims=True)
    return (row_max + np.log(np.exp(components - row_max).sum(axis=1, keepdims=True))).ravel()


def mixture_anomaly_scores(data, means, covariances, weights):
    """Anomaly score = -log p(x)."""
    return -mixture_log_density(data, means, covariances, weights)


def distribution_entropy(counts):
    """Information entropy of the ball-mass distribution, H = -sum_k p_k log p_k (0 for a single/empty ball)."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def normalize_entropy_weights(entropies, base_weight=0.1):
    """Entropies -> normalized level weights. A base value avoids zero weights; all-zero falls back to uniform."""
    weights = np.asarray(entropies, dtype=float) + base_weight
    total = float(weights.sum())
    if total <= 0:
        return np.full(weights.size, 1.0 / weights.size)
    return weights / total
