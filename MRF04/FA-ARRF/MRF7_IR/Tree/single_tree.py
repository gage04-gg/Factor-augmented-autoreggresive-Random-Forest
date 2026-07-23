import numpy as np
def dv_fun(sse, dv_pref=0.15):
    seq = np.arange(1, len(sse) + 1, dtype=float)
    dv  = 0.5 * seq**2 - seq
    dv  = dv / np.mean(dv)
    dv  = dv - dv.min() + 1.0
    return sse * (dv ** dv_pref)

def standardise(Y):
    Y   = np.asarray(Y, dtype=float)
    mu  = Y.mean(axis=0)
    sig = Y.std(axis=0, ddof=1)
    sig[sig == 0] = 1.0
    return {"Y": (Y - mu) / sig, "mean": mu, "std": sig}


class MRFTree:

    def __init__(
        self,
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
    ):
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

    def _reg_mat(self, K):
        R       = np.eye(K) * self.ridge_lambda
        R[0, 0] *= 0.01   
        return R

    def _neighbours(self, idx, n_steps, T, rando_vec, exclude=None):
        idx  = np.asarray(idx)
        nbrs = np.unique(np.concatenate([idx + n_steps, idx - n_steps]))
        excl = np.asarray(exclude) if exclude is not None else idx
        nbrs = nbrs[~np.isin(nbrs, excl)]
        nbrs = nbrs[(nbrs >= 0) & (nbrs < T)]
        if self.no_rw_trespassing:
            nbrs = np.intersect1d(nbrs, rando_vec)
        return nbrs

    def _rw_augment(self, yy, ZZ, n1, n2, full_y, full_Z):
        py, pZ = [np.atleast_1d(yy)], [np.atleast_2d(ZZ)]
        if len(n1) > 0:
            py.append(self.rw_regul    * full_y[n1])
            pZ.append(self.rw_regul    * full_Z[n1])
        if len(n2) > 0:
            py.append(self.rw_regul**2 * full_y[n2])
            pZ.append(self.rw_regul**2 * full_Z[n2])
        return np.concatenate(py), np.vstack(pZ)

    @staticmethod
    def _solve(Z, y, R):
        return np.linalg.solve(Z.T @ Z + R, Z.T @ y)

    def _block_subsample(self, T):
        n_blocks = max(1, T // self.block_size)
        groups   = np.sort(np.random.choice(n_blocks, size=T, replace=True))
        w_block  = np.random.exponential(1.0, size=n_blocks) + 0.1
        w_obs    = w_block[groups]
        thresh   = np.quantile(w_obs, 1.0 - self.subsampling_rate)
        return np.sort(np.where(w_obs > thresh)[0])

    def _sample_features(self, p):
        n   = max(1, round(p * self.mtry_frac))
        prb = np.ones(p)
        if self.trend_col_idx < p:
            prb[self.trend_col_idx] = float(self.trend_push)
        for idx in self.priority_col_idxs:
            if idx < p:
                prb[idx] = max(prb[idx], self.priority_weight)
        prb /= prb.sum()
        return np.random.choice(p, size=n, replace=False, p=prb)

    def _best_split_fast(self, ib, Xb, yb_n, Sb, R, min_leaf, p):
        feat_idx = self._sample_features(p)
        best_sse, best_split = np.inf, None
        for j in feat_idx:
            s_vals = Sb[ib, j]
            splits = np.unique(s_vals)
            if len(splits) < 2:
                continue
            sse_arr = np.full(len(splits), np.inf)
            for si, sp in enumerate(splits):
                ml = s_vals < sp;  mr = ~ml
                if ml.sum() < min_leaf or mr.sum() < min_leaf:
                    continue
                Z1, y1 = Xb[ib[ml]], yb_n[ib[ml]]
                b1 = self._solve(Z1, y1, R);  r1 = y1 - Z1 @ b1
                Z2, y2 = Xb[ib[mr]], yb_n[ib[mr]]
                b2 = self._solve(Z2, y2, R);  r2 = y2 - Z2 @ b2
                sse_arr[si] = (r1**2).sum() + (r2**2).sum()
            valid = np.isfinite(sse_arr)
            if not valid.any():
                continue
            sse_arr[valid] = dv_fun(sse_arr[valid])
            bi = int(np.nanargmin(sse_arr))
            if sse_arr[bi] < best_sse:
                best_sse   = sse_arr[bi]
                best_split = (j, splits[bi])
        return best_split

    def fit(self, X_lin, y, S_state):
        T, K = X_lin.shape
        all_data = np.column_stack([y, X_lin[:, 1:], S_state])
        std      = standardise(all_data)
        sd       = std["Y"]
        std_y    = sd[:, 0]
        std_X    = np.column_stack([np.ones(T), sd[:, 1:K]])
        std_S    = sd[:, K:]

        self._std = std;  self._K = K;  self._T = T

        rando_vec   = self._block_subsample(T)
        self._rando = rando_vec

        Xb   = std_X[rando_vec]
        yb   = std_y[rando_vec]
        Sb   = std_S[rando_vec]
        yb_n = yb + 1.5e-7 * np.random.normal(size=len(rando_vec))

        R = self._reg_mat(K)
        p = std_S.shape[1]
        min_leaf = max(
            self.min_samples_leaf,
            int(2 * self.min_leaf_frac_of_x * (K + 1) + 2)
        )
        self._min_leaf = min_leaf

        root  = dict(ib=np.arange(len(rando_vec)), ifl=rando_vec,
                     depth=0, is_leaf=False, sv=None, sp=None,
                     L=None, R=None, beta_std=None)
        stack = [root]

        while stack:
            nd         = stack.pop()
            ib, ifl, d = nd["ib"], nd["ifl"], nd["depth"]

            yy, ZZ = yb_n[ib], Xb[ib]
            n1 = self._neighbours(ifl, 1, T, rando_vec)
            n2 = np.setdiff1d(self._neighbours(ifl, 2, T, rando_vec), n1)
            yya, ZZa       = self._rw_augment(yy, ZZ, n1, n2, std_y, std_X)
            nd["beta_std"] = self._solve(ZZa, yya, R)

            if d >= self.max_depth or len(ib) < 2 * min_leaf:
                nd["is_leaf"] = True;  continue

            best = self._best_split_fast(ib, Xb, yb_n, Sb, R, min_leaf, p)

            if best is None:
                nd["is_leaf"] = True;  continue

            j_star, c_star = best
            ml = Sb[ib, j_star] < c_star;  mr = ~ml
            nd["sv"], nd["sp"] = j_star, c_star

            Lnd = dict(ib=ib[ml], ifl=ifl[ml], depth=d+1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, beta_std=None)
            Rnd = dict(ib=ib[mr], ifl=ifl[mr], depth=d+1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, beta_std=None)
            nd["L"], nd["R"] = Lnd, Rnd
            stack.extend([Lnd, Rnd])

        self._root = root
        self._beta_bank_std = np.full((T, K), np.nan)
        self._fitted_std    = np.full(T, np.nan)
        self._pred_given_tree(std_X, std_y, std_S, T, rando_vec, R)
        self._back_transform()

    def _pred_given_tree(self, std_X, std_y, std_S, T, rando_vec, R):
        leaves, stack = [], [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                nd["is_leaf"] = True;  leaves.append(nd)
            else:
                stack.extend([nd["L"], nd["R"]])

        for li, lf in enumerate(leaves):
            lf["_li"] = li
        leaf_ids = self._route_batch(std_S)

        for li, leaf in enumerate(leaves):
            ifl = leaf["ifl"]
            if len(ifl) == 0:
                continue
            yy, ZZ = std_y[ifl], std_X[ifl]
            n1 = self._neighbours(ifl, 1, T, rando_vec, exclude=ifl)
            n2 = np.setdiff1d(
                self._neighbours(ifl, 2, T, rando_vec, exclude=ifl), n1)
            yya, ZZa   = self._rw_augment(yy, ZZ, n1, n2, std_y, std_X)
            beta_hat   = self._solve(ZZa, yya, R)
            beta_final = (1.0 - self.HRW) * beta_hat + self.HRW * leaf["beta_std"]
            leaf["beta_final"] = beta_final
            mask = leaf_ids == li
            if mask.any():
                self._beta_bank_std[mask] = beta_final
                self._fitted_std[mask]    = std_X[mask] @ beta_final

    def _route_batch(self, std_S):
        T = std_S.shape[0]
        leaves, stack = [], [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                nd["is_leaf"] = True;  leaves.append(nd)
            else:
                stack.extend([nd["L"], nd["R"]])
        for li, lf in enumerate(leaves):
            lf["_li"] = li
        result = np.zeros(T, dtype=int)
        for i in range(T):
            nd = self._root
            while not nd["is_leaf"]:
                nd = nd["L"] if std_S[i, nd["sv"]] < nd["sp"] else nd["R"]
            result[i] = nd["_li"]
        return result

    def _back_transform(self):
        K = self._K;  std = self._std
        sig_y = std["std"][0];  mu_y  = std["mean"][0]
        sig_X = std["std"][1:K]; mu_X = std["mean"][1:K]
        b = self._beta_bank_std.copy()
        for k in range(1, K):
            b[:, k] = b[:, k] * sig_y / sig_X[k-1]
        b[:, 0] = b[:, 0] * sig_y + mu_y
        for k in range(1, K):
            b[:, 0] -= b[:, k] * mu_X[k-1]
        self._beta_bank = b
        self._fitted    = self._fitted_std * sig_y + mu_y

    def _leaf_beta_raw(self, leaf):
        K = self._K;  std = self._std
        sig_y = std["std"][0];  mu_y  = std["mean"][0]
        sig_X = std["std"][1:K]; mu_X = std["mean"][1:K]
        b = leaf["beta_final"].copy();  br = np.empty(K)
        for k in range(1, K):
            br[k] = b[k] * sig_y / sig_X[k-1]
        br[0] = b[0] * sig_y + mu_y
        for k in range(1, K):
            br[0] -= br[k] * mu_X[k-1]
        return br

    def _route_one_std(self, s_std):
        nd = self._root
        while not nd["is_leaf"]:
            nd = nd["L"] if s_std[nd["sv"]] < nd["sp"] else nd["R"]
        return nd

    def predict(self, X_new, S_new):
        K = self._K;  std = self._std
        s_batch = (S_new - std["mean"][K:]) / std["std"][K:]
        out = np.empty(len(X_new))
        for i in range(len(X_new)):
            leaf = self._route_one_std(s_batch[i])
            out[i] = (X_new[i] @ self._leaf_beta_raw(leaf)
                      if leaf.get("beta_final") is not None else np.nan)
        return out

    def get_betas(self, S_new):
        K = self._K;  std = self._std
        s_batch = (S_new - std["mean"][K:]) / std["std"][K:]
        out = np.full((len(S_new), K), np.nan)
        for i in range(len(S_new)):
            leaf = self._route_one_std(s_batch[i])
            if leaf.get("beta_final") is not None:
                out[i] = self._leaf_beta_raw(leaf)
        return out

    def tree_depth(self):
        max_d = [0];  stack = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                max_d[0] = max(max_d[0], d)
            else:
                stack.extend([(nd["L"], d+1), (nd["R"], d+1)])
        return max_d[0]

    def n_leaves(self):
        count = [0];  stack = [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                count[0] += 1
            else:
                stack.extend([nd["L"], nd["R"]])
        return count[0]

    def named_splits(self, col_names):
        splits = [];  stack = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if not nd["is_leaf"] and nd["sv"] is not None:
                name = (col_names[nd["sv"]] if nd["sv"] < len(col_names)
                        else f"col_{nd['sv']}")
                splits.append((d, name, nd["sp"]))
                stack.extend([(nd["L"], d+1), (nd["R"], d+1)])
        return splits
    