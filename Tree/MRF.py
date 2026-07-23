import numpy as np
from Tree.single_tree import MRFTree
class MRF:
    def __init__(self, n_trees=50, max_depth=5, min_samples_leaf=10, mtry=None, lam=1.0, zeta=0.5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.lam = lam
        self.zeta = zeta
        self.trees = []
    def block_bootstrap(self, X, y, S, block_size=8):
        T = len(y)
        indices = []
        n_blocks = int(np.ceil(T / block_size))
        for _ in range(n_blocks):
            start = np.random.randint(0, T - block_size)
            indices.extend(range(start, start + block_size))
        indices = np.array(indices[:T])
        return X[indices], y[indices], S[indices]
    def fit(self, X, y, S):
        T, p = S.shape
        if self.mtry is None:
            self.mtry = int(np.sqrt(p))
        self.trees = []
        for b in range(self.n_trees):
            X_b, y_b, S_b = self.block_bootstrap(X, y, S)
            tree = MRFTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                mtry=self.mtry,
                lam=self.lam,
                zeta=self.zeta
            )
            tree.fit(X_b, y_b, S_b)
            self.trees.append(tree)
            print(f"Tree {b+1}/{self.n_trees} trained")
    def predict(self, X, S):
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X, S)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        return np.mean(all_preds, axis=0)
    def get_betas(self, X, S):
        all_betas = []
        for tree in self.trees:
            betas_tree = []
            for i in range(len(X)):
                beta = self.get_beta_from_tree(tree.root, X[i], S[i])
                betas_tree.append(beta)
            all_betas.append(np.array(betas_tree))
        all_betas = np.array(all_betas)
        return np.mean(all_betas, axis=0)
    def get_beta_from_tree(self, node, x, s):
        if node.is_leaf:
            return node.beta
        if s[node.split_var] <= node.split_val:
            return self.get_beta_from_tree(node.left, x, s)
        else:
            return self.get_beta_from_tree(node.right, x, s)
        