import numpy as np
from joblib import Parallel, delayed
from Tree.single_tree import RF_MAF  

class RF_MAF_Forest:

    def __init__(
        self,
        n_trees=50,
        min_samples_leaf=10,
        mtry_frac=0.2,
        subsampling_rate=0.75,
        block_size=12,
        max_depth=100,
        trend_push=1,
        trend_col_idx=0,
        priority_col_idxs=None,
        priority_weight=5.0,
        n_jobs=-1,
    ):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.mtry_frac = mtry_frac
        self.subsampling_rate = subsampling_rate
        self.block_size = block_size
        self.max_depth = max_depth
        self.trend_push = trend_push
        self.trend_col_idx = trend_col_idx
        self.priority_col_idxs = priority_col_idxs or []
        self.priority_weight = priority_weight
        self.n_jobs = n_jobs
        self.trees = []

    def _make_tree(self):
        return RF_MAF(
            min_samples_leaf=self.min_samples_leaf,
            mtry_frac=self.mtry_frac,
            subsampling_rate=self.subsampling_rate,
            block_size=self.block_size,
            max_depth=self.max_depth,
            trend_push=self.trend_push,
            trend_col_idx=self.trend_col_idx,
            priority_col_idxs=self.priority_col_idxs,
            priority_weight=self.priority_weight,
        )

    @staticmethod
    def _fit_one(tree, y, S, seed):
        np.random.seed(seed)
        tree.fit(y, S)
        return tree

    def fit(self, y, S):
        T, p = S.shape
        mtry_n = max(1, round(p * self.mtry_frac))
        n_boot = int(T * self.subsampling_rate)

        print(f"    Fitting {self.n_trees} trees | T={T}, p_S={p}, "
              f"mtry≈{mtry_n}, ~bootstrap={n_boot}")

        seeds = np.random.randint(0, 100_000, size=self.n_trees)
        trees = [self._make_tree() for _ in range(self.n_trees)]

        self.trees = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._fit_one)(trees[b], y, S, seeds[b])
            for b in range(self.n_trees)
        )

    def predict(self, S):
        all_preds = np.array([t.predict(S) for t in self.trees])
        return np.nanmean(all_preds, axis=0)

    def print_diagnostics(self, S_col_names=None):
        depths = [t.tree_depth() for t in self.trees]
        n_l = [t.n_leaves() for t in self.trees]

        print(f"\n🌳 TREE DIAGNOSTICS")
        print(f"Avg depth: {np.mean(depths):.1f}  "
              f"Max: {max(depths)}  Min: {min(depths)}")
        print(f"Avg leaves: {np.mean(n_l):.1f}  "
              f"Max: {max(n_l)}  Min: {min(n_l)}")

        if S_col_names is not None:
            from collections import Counter
            all_splits = []

            for tree in self.trees:
                splits = tree.named_splits(S_col_names)
                for _, name, _ in splits:
                    all_splits.append(name)

            top = Counter(all_splits).most_common(10)

            print(f"\n📊 TOP SPLIT VARIABLES:")
            for name, cnt in top:
                print(f"{name:40s} used {cnt} times")
