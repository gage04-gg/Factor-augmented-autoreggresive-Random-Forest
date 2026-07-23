"""
MRF.py — Macroeconomic Random Forest ensemble.

Wraps B MRFTree instances.  All hyperparameter defaults mirror the paper
(Goulet Coulombe 2024, App A.6) for quarterly data.

Algorithm 1 steps 3 & 4:
  predict()   — average of tree predictions  (step 3)
  get_betas() — average of tree beta_t paths (step 4)

IMPORTANT: Every parameter here is forwarded verbatim to MRFTree.
           Adding / renaming a parameter in MRFTree must be mirrored here.
"""

import numpy as np
from joblib import Parallel, delayed
from Tree.single_tree import MRFTree


class MRF:
    """
    Macroeconomic Random Forest (FA-ARRF variant).

    Parameters — see MRFTree and paper App A.6 for descriptions.
    All defaults match the paper's quarterly setting.
    """

    def __init__(
        self,
        n_trees:            int   = 100,      # paper: ≥100 for mean prediction
        min_samples_leaf:   int   = 10,       # Minimal Node Size (quarterly)
        mtry_frac:          float = 1 / 3,    # paper default; 0.2 also robust
        ridge_lambda:       float = 0.1,      # RL (lambda)
        rw_regul:           float = 0.1,     # RWR (zeta)
        HRW:                float = 0.0,      # hierarchical blend; 0 = pure leaf
        subsampling_rate:   float = 0.75,     # Subsampling Rate
        block_size:         int   = 8,        # quarterly = 8 (2-year blocks)
        min_leaf_frac_of_x: float = 1.0,      # MLF; 1.0 OK when RWR+RL active
        no_rw_trespassing:  bool  = True,     # always True per paper
        max_depth:          int   = 100,      # effectively unlimited
        trend_push:         float = 2.0,      # push trend above 1/dim(S)
        trend_col_idx:      int   = 0,        # S_t column index of trend_t
        fast_rw:            bool  = True,     # skip RW during split search
        priority_col_idxs:  list  = None,     # leading indicator column indices
        priority_weight:    float = 5.0,      # mtry upweight for priority vars
        n_jobs:             int   = -1,       # joblib: -1 = all cores
    ):
        self.n_trees            = n_trees
        self.min_samples_leaf   = min_samples_leaf
        self.mtry_frac          = mtry_frac
        self.ridge_lambda       = ridge_lambda
        self.rw_regul           = rw_regul
        self.HRW                = HRW
        self.subsampling_rate   = subsampling_rate
        self.block_size         = block_size
        self.min_leaf_frac_of_x = min_leaf_frac_of_x
        self.no_rw_trespassing  = no_rw_trespassing
        self.max_depth          = max_depth
        self.trend_push         = float(trend_push)
        self.trend_col_idx      = trend_col_idx
        self.fast_rw            = fast_rw
        self.priority_col_idxs  = list(priority_col_idxs) if priority_col_idxs else []
        self.priority_weight    = float(priority_weight)
        self.n_jobs             = n_jobs
        self.trees: list        = []

    # ── Internal: construct a MRFTree with current settings ──────────────────
    def _make_tree(self) -> MRFTree:
        """
        All parameters forwarded to MRFTree — single place to keep in sync.
        """
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
        )

    # ── Internal: fit a single tree (picklable for joblib) ───────────────────
    @staticmethod
    def _fit_one(tree: MRFTree, X: np.ndarray, y: np.ndarray,
                 S: np.ndarray, seed: int) -> MRFTree:
        np.random.seed(seed)
        tree.fit(X, y, S)
        return tree

    # ── Public: fit B trees in parallel ──────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, S: np.ndarray):
        """
        Fit the MRF ensemble.

        Parameters
        ----------
        X : (T, K)  Linear part — [1, y_{t-1}, y_{t-2}, F1_{t-1}, F2_{t-1}]
        y : (T,)    Target (h-step ahead GDP growth, fraction not percent)
        S : (T, p)  State-space matrix (S_t) for splitting
        """
        T, p = S.shape
        K    = X.shape[1]

        # Compute effective min_leaf for display (mirrors MRFTree._min_leaf)
        eff_min_leaf = max(self.min_samples_leaf,
                           int(np.ceil(self.min_leaf_frac_of_x * K)))
        n_boot       = int(T * self.subsampling_rate)
        mtry_n       = max(1, round(p * self.mtry_frac))

        print(f"  [MRF] Fitting {self.n_trees} trees | "
              f"T={T}, p_S={p}, K={K} | "
              f"mtry≈{mtry_n} ({self.mtry_frac:.2f}×{p}), "
              f"min_leaf={eff_min_leaf}, "
              f"~n_boot={n_boot}, "
              f"fast_rw={self.fast_rw}, "
              f"block_size={self.block_size}")

        seeds = np.random.randint(0, 100_000, size=self.n_trees)
        trees = [self._make_tree() for _ in range(self.n_trees)]

        self.trees = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._fit_one)(trees[b], X, y, S, seeds[b])
            for b in range(self.n_trees)
        )

    # ── Public: OOS prediction (Algorithm 1 step 3) ───────────────────────────
    def predict(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Mean prediction across all B trees.
        NaN-safe: uses nanmean so trees with routing failures are skipped.
        """
        all_p = np.array([t.predict(X, S) for t in self.trees])
        return np.nanmean(all_p, axis=0)

    # ── Public: OOS GTVPs (Algorithm 1 step 4) ───────────────────────────────
    def get_betas(self, S: np.ndarray) -> np.ndarray:
        """
        Posterior mean of beta_t across all B trees (GTVPs).
        Returns (n, K) in original units.
        """
        all_b = np.array([t.get_betas(S) for t in self.trees])
        return np.nanmean(all_b, axis=0)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def print_diagnostics(self, S_col_names: list = None):
        """
        Print average/max/min tree depth and leaf count.
        If S_col_names provided, also print top split variables across ALL trees
        and the first 12 splits of Tree 0.
        """
        if not self.trees:
            print("  [MRF] No trees fitted yet.")
            return

        depths = [t.tree_depth() for t in self.trees]
        n_l    = [t.n_leaves()   for t in self.trees]
        print(f"\n  TREE DIAGNOSTICS ({len(self.trees)} trees)")
        print(f"  Depth  — avg: {np.mean(depths):.1f}  "
              f"max: {max(depths)}  min: {min(depths)}")
        print(f"  Leaves — avg: {np.mean(n_l):.1f}  "
              f"max: {max(n_l)}  min: {min(n_l)}")

        if S_col_names is not None:
            from collections import Counter
            # Top split variables across ALL trees
            all_splits = []
            for tree in self.trees:
                for _, name, _ in tree.named_splits(S_col_names):
                    all_splits.append(name)
            top = Counter(all_splits).most_common(15)
            print(f"\n  TOP SPLIT VARIABLES (all {len(self.trees)} trees):")
            for name, cnt in top:
                print(f"    {name:45s}  {cnt:4d} uses")

            # First splits of Tree 0
            splits_0 = sorted(self.trees[0].named_splits(S_col_names),
                               key=lambda x: x[0])
            print(f"\n  TREE 0 SPLITS (first 12 by depth):")
            for depth, name, thresh in splits_0[:12]:
                print(f"    depth {depth:2d}: {name:40s} < {thresh:.4f}")
