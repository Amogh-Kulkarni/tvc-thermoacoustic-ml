"""
Nonlinear prediction error via Sugihara's simplex projection.

Sugihara & May, "Nonlinear forecasting as a way of distinguishing chaos
from measurement error in time series" Nature 344 (1990): 734-741.

Algorithm:
    1. Subsample the signal
    2. Time-delay embed in dimension m with lag tau
    3. Split into library (first 70% of embedded points) and
       prediction set (remainder, excluding the last max_h points
       so that truth values exist for all horizons)
    4. For each prediction point q, find the n_neighbors nearest
       library points in embedding space
    5. For each horizon h, predict x(q + h) as an exponentially
       distance-weighted mean of the neighbors' values h steps ahead
    6. Report normalized RMSE = sqrt(mean(err^2)) / std(x)

Returns errors at horizons 1, 5, 10, 20, 50 samples plus the
growth ratio err(h=50) / err(h=1).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from existing_methods import estimate_delay_from_autocorr


def compute_prediction_error(x, fs=20000, emb_dim=5, delay=None,
                                n_neighbors=4, horizons=(1, 5, 10, 20, 50),
                                subsample_factor=20):
    """Sugihara simplex projection prediction error."""
    horizons = list(horizons)
    nan_out = {
        'horizons': horizons,
        'pred_errors': [float('nan')] * len(horizons),
        'error_h1': float('nan'),
        'error_h50': float('nan'),
        'error_growth_ratio': float('nan'),
    }

    x_sub = x[::subsample_factor].astype(np.float64)
    N = len(x_sub)

    if delay is None:
        delay = estimate_delay_from_autocorr(x_sub)
    delay = max(1, int(delay))

    max_h = max(horizons)
    n_points = N - (emb_dim - 1) * delay
    if n_points < 100 + max_h:
        return nan_out

    embedded = np.zeros((n_points, emb_dim))
    for d in range(emb_dim):
        embedded[:, d] = x_sub[d * delay : d * delay + n_points]

    # Library / prediction split. Library goes up to lib_end-1.
    # Ensure that every library point j has j + max_h < n_points, so it
    # can be used as a neighbor at any horizon without indexing past the end.
    lib_end_cap = n_points - max_h - 1
    lib_end = min(int(0.7 * n_points), lib_end_cap)
    if lib_end < n_neighbors + 1:
        return nan_out

    # Prediction query range: queries need truth available at max horizon
    pred_start = lib_end
    pred_end = n_points - max_h
    if pred_end - pred_start < 10:
        return nan_out

    library = embedded[:lib_end]
    pred_query = embedded[pred_start:pred_end]

    # Find nearest neighbors in the library for each query point
    try:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(library)
        dists, idxs = nbrs.kneighbors(pred_query)
    except Exception:
        return nan_out

    # Sugihara weights: w_k ~ exp(-d_k / d_min)
    d_min = np.maximum(dists[:, 0:1], 1e-12)
    weights = np.exp(-dists / d_min)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Normalize errors by signal std
    x_std = float(np.std(x_sub))
    if x_std < 1e-12:
        x_std = 1.0

    query_abs_idx = np.arange(pred_start, pred_end)

    pred_errors = []
    for h in horizons:
        target_idxs = query_abs_idx + h
        if target_idxs.max() >= n_points:
            pred_errors.append(float('nan'))
            continue
        targets = embedded[target_idxs, 0]  # true x(t+h)

        nn_future = idxs + h  # shape (n_query, n_neighbors)
        if nn_future.max() >= n_points:
            # Library cap should prevent this, but filter defensively
            keep = (nn_future < n_points).all(axis=1)
            if keep.sum() < 5:
                pred_errors.append(float('nan'))
                continue
            nn_future = nn_future[keep]
            w = weights[keep]
            t = targets[keep]
        else:
            w = weights
            t = targets

        nn_vals = embedded[nn_future, 0]  # (n_query, n_neighbors)
        preds = np.sum(w * nn_vals, axis=1)
        err = float(np.sqrt(np.mean((preds - t) ** 2)) / x_std)
        pred_errors.append(err)

    h1 = pred_errors[0] if horizons[0] == 1 else pred_errors[0]
    h50 = pred_errors[-1] if horizons[-1] == 50 else pred_errors[-1]
    if h1 and not np.isnan(h1) and h1 > 0 and not np.isnan(h50):
        growth = float(h50 / h1)
    else:
        growth = float('nan')

    return {
        'horizons': horizons,
        'pred_errors': pred_errors,
        'error_h1': h1,
        'error_h50': h50,
        'error_growth_ratio': growth,
    }
