import numpy as np
from joblib import Parallel, delayed
from Tree.single_tree import MRFTree
from collections import Counter

class MRF:
    def __init__(
        self,
        n_trees:            int   = 50,
        min_samples_leaf:   int   = 10,
        mtry_frac:          float = 1/3,
        ridge_lambda:       float = 0.1,
        rw_regul:           float = 0.75,
        HRW:                float = 0.0,
        subsampling_rate:   float = 0.75,
        block_size:         int   = 8,
        min_leaf_frac_of_x: float = 1.0,
        no_rw_trespassing:  bool  = True,
        max_depth:          int   = 100,
        trend_push:         int   = 2,
        trend_col_idx:      int   = 0,
        fast_rw:            bool  = True,
        priority_col_idxs:  list  = None,
        priority_weight:    float = 5.0,
        n_jobs:             int   = -1,
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
        self.trend_push         = trend_push
        self.trend_col_idx      = trend_col_idx
        self.fast_rw            = fast_rw
        self.priority_col_idxs  = priority_col_idxs or []
        self.priority_weight    = priority_weight
        self.n_jobs             = n_jobs
        self.trees              = []

    def _make_tree(self):
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

    @staticmethod
    def _fit_one(tree, X, y, S, seed):
        np.random.seed(seed)
        tree.fit(X, y, S)
        return tree

    def fit(self, X, y, S):
        T, p = S.shape;  K = X.shape[1]
        ml = max(self.min_samples_leaf,
                 int(2 * self.min_leaf_frac_of_x * (K + 1) + 2))
        mtry_n = max(1, round(p * self.mtry_frac))
        n_boot = int(T * self.subsampling_rate)
        print(f"    Fitting {self.n_trees} trees | T={T}, p_S={p}, "
              f"mtry≈{mtry_n}, min_leaf={ml}, ~bootstrap={n_boot}")
        seeds = np.random.randint(0, 100_000, size=self.n_trees)
        trees = [self._make_tree() for _ in range(self.n_trees)]
        self.trees = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._fit_one)(trees[b], X, y, S, seeds[b])
            for b in range(self.n_trees)
        )
    def print_diagnostics(self, S_col_names=None):
        depths = [t.tree_depth() for t in self.trees]
        n_l    = [t.n_leaves()   for t in self.trees]
        print(f"\n  🌳 TREE DIAGNOSTICS")
        print(f"  Avg depth: {np.mean(depths):.1f}  "
              f"Max: {max(depths)}  Min: {min(depths)}")
        print(f"  Avg leaves: {np.mean(n_l):.1f}  "
              f"Max: {max(n_l)}  Min: {min(n_l)}")
        if S_col_names:
            splits0 = self.trees[0].named_splits(S_col_names)
            print(f"\n  🔍 NAMED SPLITS (Tree 0):")
            for d, name, thresh in sorted(splits0, key=lambda x: x[0])[:10]:
                print(f"    depth {d}: {name:40s} < {thresh:.4f}")
            all_splits = []
            for tree in self.trees:
                for _, name, _ in tree.named_splits(S_col_names):
                    all_splits.append(name)
            print(f"\n  📊 TOP SPLIT VARIABLES:")
            for name, cnt in Counter(all_splits).most_common(10):
                print(f"    {name:45s} used {cnt:3d}×")

    def predict(self, X, S):
        return np.nanmean([t.predict(X, S) for t in self.trees], axis=0)

    def get_betas(self, S):
        return np.nanmean([t.get_betas(S) for t in self.trees], axis=0)
    