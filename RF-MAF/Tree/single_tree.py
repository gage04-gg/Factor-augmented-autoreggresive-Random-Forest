import numpy as np

class RF_MAF:

    def __init__(
        self,
        min_samples_leaf=10,
        mtry_frac=0.2,
        subsampling_rate=0.75,
        block_size=12,
        max_depth=100,
        trend_push=1,
        trend_col_idx=0,
        priority_col_idxs=None,
        priority_weight=5.0,
    ):
        self.min_samples_leaf = min_samples_leaf
        self.mtry_frac = mtry_frac
        self.subsampling_rate = subsampling_rate
        self.block_size = block_size
        self.max_depth = max_depth
        self.trend_push = trend_push
        self.trend_col_idx = trend_col_idx
        self.priority_col_idxs = priority_col_idxs or []
        self.priority_weight = priority_weight

    # ── Standardisation ──────────────────────────────────────────────────────
    def _standardise(self, Y):
        Y = np.asarray(Y, dtype=float)
        mu = Y.mean(axis=0)
        sig = Y.std(axis=0, ddof=1)
        sig[sig == 0] = 1.0
        return {"Y": (Y - mu) / sig, "mean": mu, "std": sig}

    # ── Block subsampling ────────────────────────────────────────────────────
    def _block_subsample(self, T):
        n_blocks = max(1, T // self.block_size)
        groups = np.sort(np.random.choice(n_blocks, size=T, replace=True))
        w_block = np.random.exponential(1.0, size=n_blocks) + 0.1
        w_obs = w_block[groups]
        thresh = np.quantile(w_obs, 1.0 - self.subsampling_rate)
        return np.sort(np.where(w_obs > thresh)[0])

    # ── Feature sampling (mtry) ──────────────────────────────────────────────
    def _sample_features(self, p):
        n = max(1, round(p * self.mtry_frac))
        prb = np.ones(p)

        # Trend priority
        if self.trend_col_idx < p:
            prb[self.trend_col_idx] = float(self.trend_push)

        # Priority variables
        for idx in self.priority_col_idxs:
            if idx < p:
                prb[idx] = max(prb[idx], self.priority_weight)

        prb /= prb.sum()
        return np.random.choice(p, size=n, replace=False, p=prb)

    # ── Best split (CORE RF CHANGE: mean instead of regression) ──────────────
    def _best_split(self, ib, yb, Sb, min_leaf, p):
        feat_idx = self._sample_features(p)
        best_sse, best_split = np.inf, None

        for j in feat_idx:
            s_vals = Sb[ib, j]
            splits = np.unique(s_vals)

            if len(splits) < 2:
                continue

            for sp in splits:
                ml = s_vals < sp
                mr = ~ml

                if ml.sum() < min_leaf or mr.sum() < min_leaf:
                    continue

                y1 = yb[ib[ml]]
                y2 = yb[ib[mr]]

                # 🔥 KEY CHANGE: leaf mean instead of regression
                mean1 = y1.mean()
                mean2 = y2.mean()

                sse = ((y1 - mean1) ** 2).sum() + ((y2 - mean2) ** 2).sum()

                if sse < best_sse:
                    best_sse = sse
                    best_split = (j, sp)

        return best_split

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(self, y, S_state):
        T = len(y)

        # Standardise
        all_data = np.column_stack([y, S_state])
        std = self._standardise(all_data)

        sd = std["Y"]
        std_y = sd[:, 0]
        std_S = sd[:, 1:]

        self._std = std
        self._T = T

        # Subsample
        rando_vec = self._block_subsample(T)
        self._rando = rando_vec

        yb = std_y[rando_vec]
        Sb = std_S[rando_vec]

        # Small noise (as in your code)
        yb_n = yb + 1e-7 * np.random.normal(size=len(yb))

        p = std_S.shape[1]
        min_leaf = self.min_samples_leaf

        # ── Build tree ───────────────────────────────────────────────────────
        root = dict(
            ib=np.arange(len(rando_vec)),
            depth=0,
            is_leaf=False,
            sv=None,
            sp=None,
            L=None,
            R=None,
            y_mean=None
        )

        stack = [root]

        while stack:
            nd = stack.pop()
            ib, d = nd["ib"], nd["depth"]

            # Leaf value
            nd["y_mean"] = yb_n[ib].mean()

            # Stop condition
            if d >= self.max_depth or len(ib) < 2 * min_leaf:
                nd["is_leaf"] = True
                continue

            best = self._best_split(ib, yb_n, Sb, min_leaf, p)

            if best is None:
                nd["is_leaf"] = True
                continue

            j_star, c_star = best

            ml = Sb[ib, j_star] < c_star
            mr = ~ml

            nd["sv"], nd["sp"] = j_star, c_star

            Lnd = dict(ib=ib[ml], depth=d+1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, y_mean=None)

            Rnd = dict(ib=ib[mr], depth=d+1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, y_mean=None)

            nd["L"], nd["R"] = Lnd, Rnd
            stack.extend([Lnd, Rnd])

        self._root = root

    # ── Routing ─────────────────────────────────────────────────────────────
    def _route_one(self, s):
        nd = self._root
        while not nd["is_leaf"]:
            nd = nd["L"] if s[nd["sv"]] < nd["sp"] else nd["R"]
        return nd

    # ── Predict ─────────────────────────────────────────────────────────────
    def predict(self, S_new):
        std = self._std
        K = 1  # y column

        s_batch = (S_new - std["mean"][K:]) / std["std"][K:]

        out = np.empty(len(S_new))

        for i in range(len(S_new)):
            leaf = self._route_one(s_batch[i])
            out[i] = leaf["y_mean"]

        # Back transform y
        sig_y = std["std"][0]
        mu_y = std["mean"][0]

        return out * sig_y + mu_y

    # ── Diagnostics ─────────────────────────────────────────────────────────
    def tree_depth(self):
        max_d = [0]
        stack = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if nd["is_leaf"]:
                max_d[0] = max(max_d[0], d)
            else:
                stack.extend([(nd["L"], d+1), (nd["R"], d+1)])
        return max_d[0]

    def n_leaves(self):
        count = 0
        stack = [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"]:
                count += 1
            else:
                stack.extend([nd["L"], nd["R"]])
        return count
    