import numpy as np
class MRFNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.split_var = None
        self.split_val = None
        self.beta = None
        self.is_leaf = True
class MRFTree:
    def __init__(self, max_depth=5, min_samples_leaf=10, mtry=10, lam=1.0, zeta=0.5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.lam = lam
        self.zeta = zeta
    def compute_weights(self, indices, T):
        w = np.zeros(T)
        for t in indices:
            w[t] = 1
            if t-1 >= 0: w[t-1] = max(w[t-1], self.zeta)
            if t+1 < T: w[t+1] = max(w[t+1], self.zeta)
            if t-2 >= 0: w[t-2] = max(w[t-2], self.zeta**2)
            if t+2 < T: w[t+2] = max(w[t+2], self.zeta**2)
        return w
    def fit_beta(self, X, y, weights):
        W = np.diag(weights)
        XtW = X.T @ W
        beta = np.linalg.inv(XtW @ X + self.lam * np.eye(X.shape[1])) @ XtW @ y
        return beta
    def compute_rss(self, X, y, beta, weights):
        residuals = y - X @ beta
        return np.sum(weights * (residuals ** 2))
    def find_best_split(self, X, y, S, indices):
        T, p = S.shape
        best_loss = np.inf
        best_split = None
        features = np.random.choice(p, self.mtry, replace=False)
        for j in features:
            values = S[indices, j]
            thresholds = np.unique(values)
            for c in thresholds:
                left_idx = indices[values <= c]
                right_idx = indices[values > c]
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                w_left = self.compute_weights(left_idx, T)
                w_right = self.compute_weights(right_idx, T)
                beta_left = self.fit_beta(X, y, w_left)
                beta_right = self.fit_beta(X, y, w_right)
                loss_left = self.compute_rss(X, y, beta_left, w_left)
                loss_right = self.compute_rss(X, y, beta_right, w_right)
                loss = loss_left + loss_right
                if loss < best_loss:
                    best_loss = loss
                    best_split = (j, c, left_idx, right_idx)
        return best_split
    def build_tree(self, X, y, S, indices, depth):
        node = MRFNode()
        T = len(y)
        weights = self.compute_weights(indices, T)
        node.beta = self.fit_beta(X, y, weights)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        split = self.find_best_split(X, y, S, indices)
        if split is None:
            return node
        j, c, left_idx, right_idx = split
        node.is_leaf = False
        node.split_var = j
        node.split_val = c
        node.left = self.build_tree(X, y, S, left_idx, depth + 1)
        node.right = self.build_tree(X, y, S, right_idx, depth + 1)
        return node
    def fit(self, X, y, S):
        indices = np.arange(len(y))
        self.root = self.build_tree(X, y, S, indices, 0)
    def predict_one(self, x, s, node):
        if node.is_leaf:
            return x @ node.beta
        if s[node.split_var] <= node.split_val:
            return self.predict_one(x, s, node.left)
        else:
            return self.predict_one(x, s, node.right)
    def predict(self, X, S):
        preds = []
        for i in range(len(X)):
            preds.append(self.predict_one(X[i], S[i], self.root))
        return np.array(preds)