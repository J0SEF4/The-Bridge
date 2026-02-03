#!/usr/bin/env python3
"""
Compute intrinsic dimension per identity using graph geodesic distances
and the Granata et al. approach (compare distance distribution to m-hypersphere).

Outputs: experiments/dim estimation/intrinsic_dim_per_identity.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm


def build_knn_graph(X, k, metric='euclidean'):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)
    n = X.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j_idx, dist in zip(indices[i], distances[i]):
            rows.append(i)
            cols.append(j_idx)
            data.append(dist)
    # make symmetric: take min of both directions via undirected edges
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.minimum(A.T)
    return A


def geodesic_distances_from_graph(A):
    # compute all-pairs shortest path distances
    D = shortest_path(A, directed=False, unweighted=False)
    # shortest_path returns inf for disconnected pairs; remove infs
    D = np.array(D)
    D = D[np.isfinite(D)]
    return D


def estimate_density(distances, grid=None, bandwidth=None):
    # distances: 1D array of pairwise geodesic distances
    if grid is None:
        grid = np.linspace(distances.min(), distances.max(), 256)
    if bandwidth is None:
        # rule-of-thumb bandwidth
        bw = 1.06 * distances.std() * (len(distances) ** (-1/5)) if distances.std() > 0 else 1e-3
    else:
        bw = bandwidth
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(distances.reshape(-1, 1))
    logp = kde.score_samples(grid.reshape(-1, 1))
    p = np.exp(logp)
    return grid, p


def fit_m_from_distribution(grid, p):
    # follow Granata: consider r in [rmax - 2*sigma, rmax]
    rmax_idx = np.argmax(p)
    rmax = grid[rmax_idx]
    sigma = np.sqrt(np.sum((grid - np.sum(grid * p)) ** 2 * p) / np.sum(p))
    # approximate sigma of distribution using second moment (coarse)
    lower = rmax - 2 * sigma
    mask = (grid >= max(grid.min(), lower)) & (grid <= rmax)
    if mask.sum() < 5:
        mask = grid <= rmax
    r_sel = grid[mask]
    p_sel = p[mask]
    # normalize p_sel by p(rmax)
    p_sel = p_sel / (p[rmax_idx] + 1e-12)
    # avoid zeros
    p_sel = np.clip(p_sel, 1e-12, None)
    y = np.log(p_sel)
    # model: log p(r)/p(rmax) = (m-1) * log(sin(pi*r/(2*rmax)))
    x_arg = np.sin(np.pi * r_sel / (2 * rmax))
    # numerical safety
    x_arg = np.clip(x_arg, 1e-12, 1 - 1e-12)
    x = np.log(x_arg)
    # linear fit y = (m-1) * x  -> slope = m-1
    A = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = sol[0]
    m_est = float(slope + 1.0)
    # compute RMSE on y
    y_pred = slope * x + sol[1]
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return m_est, rmse


def estimate_id_for_cluster(X, k_candidates=(4, 7, 9, 15), metric='cosine'):
    # X: n x d embeddings
    if len(X) < 5:
        return {'n': len(X), 'm': np.nan, 'm_rounded': None, 'k': None, 'rmse': None}
    best = None
    for k in k_candidates:
        k_use = min(k, len(X) - 1)
        try:
            A = build_knn_graph(X, k_use, metric=metric)
            D = shortest_path(A, directed=False, unweighted=False)
            # collect upper triangle finite distances
            D = np.array(D)
            mask = np.isfinite(D)
            if not mask.any():
                continue
            # take only pairwise distances (i<j)
            iu = np.triu_indices(D.shape[0], k=1)
            dij = D[iu]
            dij = dij[np.isfinite(dij)]
            if len(dij) < 20:
                continue
            grid, p = estimate_density(dij)
            m_est, rmse = fit_m_from_distribution(grid, p)
            if best is None or (rmse < best['rmse']):
                best = {'k': k_use, 'm': m_est, 'rmse': rmse}
        except Exception:
            continue
    if best is None:
        return {'n': len(X), 'm': np.nan, 'm_rounded': None, 'k': None, 'rmse': None}
    return {'n': len(X), 'm': best['m'], 'm_rounded': int(round(best['m'])), 'k': best['k'], 'rmse': best['rmse']}


def main(csv_path='../frgc cluster/frgc_face_embeddings.csv', out_csv='intrinsic_dim_per_identity.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV embeddings not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # assume embeddings columns are all except image_file and person_id
    drop_cols = ['image_file', 'person_id']
    emb_cols = [c for c in df.columns if c not in drop_cols]
    results = []
    grouped = df.groupby('person_id')
    for pid, g in tqdm(grouped, desc='identities'):
        X = g[emb_cols].values
        # if embeddings are normalized to unit sphere, metric='cosine' is fine
        res = estimate_id_for_cluster(X, k_candidates=(4,7,9,15), metric='cosine')
        res.update({'person_id': pid})
        results.append(res)
    out_df = pd.DataFrame(results)
    out_df = out_df[['person_id', 'n', 'k', 'm', 'm_rounded', 'rmse']]
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote results to {out_csv}")


if __name__ == '__main__':
    main()
