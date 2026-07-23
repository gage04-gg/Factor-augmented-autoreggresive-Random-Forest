"""
MRF – Macroeconomic Random Forest (ensemble)
============================================
All defaults match single_tree.py (paper App A.6, quarterly).

Key parameters for GTVP coherence (paper Figure 22):
  - fast_rw=True        skip RW during split search; still used at leaf estimation
  - rw_regul=0.85       strengthened temporal anchor (was 0.75)
  - ridge_lambda=0.1    base penalty; AR lags get ×10, factors get ×0.4 (in _reg_mat)
  - persistence_thresh  φ₁+φ₂ ceiling (default 0.3); if exceeded → shrink AR coeffs
  - persistence_shrink  multiplicative factor applied to phi1/phi2 (default 0.65)

These settings work together to ensure the persistence SUM φ₁+φ₂ falls during
recessions rather than rising, matching paper Figure 22.
"""

import numpy as np
from joblib import Parallel, delayed
from Tree.single_tree import MRFTree


class MRF:

    def __init__(
        self,
        n_trees:             int   = 100,
        min_samples_leaf:    int   = 10,
        mtry_frac:           float = 0.2,
        ridge_lambda:        float = 0.1,
        rw_regul:            float = 0.85,   # strengthened temporal anchor
        HRW:                 float = 0.0,
        subsampling_rate:    float = 0.75,
        block_size:          int   = 8,
        min_leaf_frac_of_x:  float = 0.0,
        no_rw_trespassing:   bool  = True,
        max_depth:           int   = 100,
        trend_push:          int   = 2,
        trend_col_idx:       int   = 0,
        fast_rw:             bool  = True,   # fast split search; leaf RW still on
        priority_col_idxs:   list  = None,
        priority_weight:     float = 5.0,
        ar_ridge_mult:       float = 10.0,
        factor_ridge_mult:   float = 0.4,
        persistence_thresh:  float = 0.07,    # φ₁+φ₂ ceiling before shrinkage fires
        persistence_shrink:  float = 0.65,   # multiplicative shrink on phi1, phi2
        n_jobs:              int   = -1,
    ):
        self.n_trees             = n_trees
        self.min_samples_leaf    = min_samples_leaf
        self.mtry_frac           = mtry_frac
        self.ridge_lambda        = ridge_lambda
        self.rw_regul            = rw_regul
        self.HRW                 = HRW
        self.subsampling_rate    = subsampling_rate
        self.block_size          = block_size
        self.min_leaf_frac_of_x  = min_leaf_frac_of_x
        self.no_rw_trespassing   = no_rw_trespassing
        self.max_depth           = max_depth
        self.trend_push          = trend_push
        self.trend_col_idx       = trend_col_idx
        self.fast_rw             = fast_rw
        self.priority_col_idxs   = priority_col_idxs or []
        self.priority_weight     = priority_weight
        self.ar_ridge_mult       = ar_ridge_mult
        self.factor_ridge_mult   = factor_ridge_mult
        self.persistence_thresh  = persistence_thresh
        self.persistence_shrink  = persistence_shrink
        self.n_jobs              = n_jobs
        self.trees               = []

    def _make_tree(self) -> MRFTree:
        return MRFTree(
            min_samples_leaf   = self.min_samples_leaf,
            mtry_frac          = self.mtry_frac,
            ridge_lambda       = self.ridge_lambda,
            rw_regul           = self.rw_regul,
            HRW                = self.HRW,
            subsampling_rate   = self.subsampling_rate,
            block_size         = self.block_size,
            min_leaf_frac_of_x = self.min_leaf_frac_of_x,
            no_rw_trespassing  = self.no_rw_trespassing,
            max_depth          = self.max_depth,
            trend_push         = self.trend_push,
            trend_col_idx      = self.trend_col_idx,
            fast_rw            = self.fast_rw,
            priority_col_idxs  = self.priority_col_idxs,
            priority_weight    = self.priority_weight,
            ar_ridge_mult      = self.ar_ridge_mult,
            factor_ridge_mult  = self.factor_ridge_mult,
            persistence_thresh = self.persistence_thresh,
            persistence_shrink = self.persistence_shrink,
        )

    @staticmethod
    def _fit_one(tree, X, y, S, seed):
        np.random.seed(seed)
        tree.fit(X, y, S)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray, S: np.ndarray):
        T, p = S.shape
        K    = X.shape[1]
        mtry_n   = max(1, round(p * self.mtry_frac))
        min_leaf = max(self.min_samples_leaf,
                       int(2 * self.min_leaf_frac_of_x * (K + 1) + 2))
        n_boot   = int(T * self.subsampling_rate)
        print(f"    Fitting {self.n_trees} trees | T={T}, p_S={p}, "
              f"mtry≈{mtry_n}, min_leaf={min_leaf}, "
              f"~boot={n_boot}, fast_rw={self.fast_rw}, "
              f"rw_regul={self.rw_regul}, "
              f"persist_thresh={self.persistence_thresh}, "
              f"persist_shrink={self.persistence_shrink}")

        seeds = np.random.randint(0, 100_000, size=self.n_trees)
        trees = [self._make_tree() for _ in range(self.n_trees)]
        if self.n_jobs == 1:
            self.trees = [
                self._fit_one(trees[b], X, y, S, seeds[b])
                for b in range(self.n_trees)
            ]
        else:
            self.trees = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self._fit_one)(trees[b], X, y, S, seeds[b])
                for b in range(self.n_trees)
            )

    def print_diagnostics(self, S_col_names: list = None):
        depths = [t.tree_depth() for t in self.trees]
        n_l    = [t.n_leaves()   for t in self.trees]
        print(f"\n  🌳 TREE DIAGNOSTICS  ({len(self.trees)} trees)")
        print(f"  Avg depth : {np.mean(depths):.1f}  "
              f"Max: {max(depths)}  Min: {min(depths)}")
        print(f"  Avg leaves: {np.mean(n_l):.1f}  "
              f"Max: {max(n_l)}  Min: {min(n_l)}")
        if S_col_names is not None:
            splits = self.trees[0].named_splits(S_col_names)
            print(f"\n  🔍 NAMED SPLITS (Tree 0, {len(splits)} total):")
            for depth, name, thresh in sorted(splits, key=lambda x: x[0])[:12]:
                print(f"    depth {depth}: {name:40s} < {thresh:.4f}")
            from collections import Counter
            all_splits = []
            for tree in self.trees:
                for _, name, _ in tree.named_splits(S_col_names):
                    all_splits.append(name)
            top = Counter(all_splits).most_common(12)
            print(f"\n  📊 TOP SPLIT VARIABLES across all trees:")
            for name, cnt in top:
                print(f"    {name:45s}  used {cnt:4d} times")

    def predict(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        all_p = np.array([t.predict(X, S) for t in self.trees])
        return np.nanmean(all_p, axis=0)

    def get_betas(self, S: np.ndarray) -> np.ndarray:
        """Posterior mean GTVPs averaged across all trees."""
        all_b = np.array([t.get_betas(S) for t in self.trees])
        return np.nanmean(all_b, axis=0)

    def get_beta_quantiles(self, S: np.ndarray,
                           q: tuple = (0.10, 0.32, 0.68, 0.90)) -> np.ndarray:
        """
        Credible bands for GTVPs across the tree distribution.
        Returns array of shape (len(q), n_obs, K).
        Matches the grey bands in paper Figures 22-23.
        """
        all_b = np.array([t.get_betas(S) for t in self.trees])
        return np.nanquantile(all_b, q, axis=0)
    
