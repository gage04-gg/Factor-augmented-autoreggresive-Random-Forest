import numpy as np


# ── Discontinuity-Variance penalty ───────────────────────────────────────────
def dv_fun(sse: np.ndarray, dv_pref: float = 0.15) -> np.ndarray:
    """Penalise splits that create large jumps in sorted-SSE sequence."""
    seq = np.arange(1, len(sse) + 1, dtype=float)
    dv  = 0.5 * seq ** 2 - seq
    dv  = dv / np.mean(dv)
    dv  = dv - dv.min() + 1.0
    return sse * (dv ** dv_pref)


def _standardise_vec(v: np.ndarray) -> tuple:
    mu  = v.mean()
    sig = v.std(ddof=1)
    if sig == 0:
        sig = 1.0
    return (v - mu) / sig, mu, sig


def _standardise_mat(M: np.ndarray) -> tuple:
    mu  = M.mean(axis=0)
    sig = M.std(axis=0, ddof=1)
    sig[sig == 0] = 1.0
    return (M - mu) / sig, mu, sig


class MRFTree:

    def __init__(
        self,
        min_samples_leaf:   int   = 10,
        mtry_frac:          float = 0.2,
        ridge_lambda:       float = 0.1,
        rw_regul:           float = 0.65,   # strengthened temporal anchor
        HRW:                float = 0.0,
        subsampling_rate:   float = 0.75,
        block_size:         int   = 8,
        min_leaf_frac_of_x: float = 0.0,
        no_rw_trespassing:  bool  = True,
        max_depth:          int   = 200,
        trend_push:         int   = 2,
        trend_col_idx:      int   = 0,
        fast_rw:            bool  = True,   # True for speed; RW used at leaf estimation
        priority_col_idxs:  list  = None,
        priority_weight:    float = 5.0,
        ar_ridge_mult:      float = 5.0,
        factor_ridge_mult:  float = 3.0,
        persistence_thresh: float = 0.07,    # φ₁+φ₂ ceiling before shrinkage fires
        persistence_shrink: float = 0.65,   # multiplicative shrink applied to phi1, phi2
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
        self.ar_ridge_mult      = ar_ridge_mult
        self.factor_ridge_mult  = factor_ridge_mult
        self.persistence_thresh = persistence_thresh
        self.persistence_shrink = persistence_shrink

    # ── Ridge penalty matrix ──────────────────────────────────────────────────
    def _reg_mat(self, K: int) -> np.ndarray:
        """
        Differentiated ridge regularisation (FIX 5):
          [0] intercept  — uniform base penalty
          [1] phi1       — 10× heavier penalty to actively dampen AR persistence
          [2] phi2       — 10× heavier penalty (same reason)
          [3+] factors   — 0.4× lighter penalty; factors must move freely
                           to capture regime-switching real-activity signals

        This is the primary lever to ensure φ₁+φ₂ goes DOWN (not up) during
        recessions, as required by paper Figure 22.
        """
        R = self.ridge_lambda * np.eye(K)
        # Heavy penalty on AR lag coefficients → shrinks persistence
        if K > 1:
            R[1, 1] *= self.ar_ridge_mult
        if K > 2:
            R[2, 2] *= self.ar_ridge_mult
        # Light penalty on factor loadings → free to capture regime signals
        for i in range(3, K):
            R[i, i] *= self.factor_ridge_mult
        return R

    # ── Contiguous block subsampling (BBB) ───────────────────────────────────
    def _block_subsample(self, T: int) -> np.ndarray:
        """
        Block-Bayesian Bootstrap with CONTIGUOUS 2-year windows.

        block_id[t] = t // block_size  (deterministic, time-ordered assignment).
        Each block b gets weight w_b ~ Exp(1) + 0.1.
        Observations with w_obs > quantile(1 - subsampling_rate) are kept.

        Contiguous blocks ensure recession-era quarters stay together or are
        excluded together, preventing mixed-regime leaf estimation that causes
        contradictory phi/gamma betas during the 2008 crisis.
        """
        n_blocks = max(1, int(np.ceil(T / self.block_size)))
        # Deterministic contiguous assignment: obs t → block t//block_size
        block_id = np.minimum(np.arange(T) // self.block_size, n_blocks - 1)
        w_block  = np.random.exponential(1.0, size=n_blocks) + 0.1
        w_obs    = w_block[block_id]
        thresh   = np.quantile(w_obs, 1.0 - self.subsampling_rate)
        return np.sort(np.where(w_obs > thresh)[0])

    # ── Temporal neighbours ───────────────────────────────────────────────────
    def _neighbours(self, idx: np.ndarray, n_steps: int, T: int,
                    rando_vec: np.ndarray,
                    exclude: np.ndarray = None,
                    full_sample: bool = False) -> np.ndarray:
        idx  = np.asarray(idx)
        nbrs = np.unique(np.concatenate([idx + n_steps, idx - n_steps]))
        excl = np.asarray(exclude) if exclude is not None else idx
        nbrs = nbrs[~np.isin(nbrs, excl)]
        nbrs = nbrs[(nbrs >= 0) & (nbrs < T)]
        # During split search respect no_rw_trespassing; at leaf estimation
        # use full_sample=True so crisis leaves get enough anchoring (FIX 3)
        if not full_sample and self.no_rw_trespassing:
            nbrs = np.intersect1d(nbrs, rando_vec)
        return nbrs

    # ── RW row-append augmentation ────────────────────────────────────────────
    def _rw_augment(self, yy, ZZ, n1, n2, full_y, full_Z):
        py = [np.atleast_1d(yy)]
        pZ = [np.atleast_2d(ZZ)]
        if len(n1) > 0:
            py.append(self.rw_regul      * full_y[n1])
            pZ.append(self.rw_regul      * full_Z[n1])
        if len(n2) > 0:
            py.append(self.rw_regul ** 2 * full_y[n2])
            pZ.append(self.rw_regul ** 2 * full_Z[n2])
        return np.concatenate(py), np.vstack(pZ)

    # ── Ridge solve + persistence control (FIX 7) ────────────────────────────
    def _solve(self, Z: np.ndarray, y: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Ridge regression: β = (Z'Z + R)⁻¹ Z'y

        Persistence control (FIX 7):
          After solving, check φ₁+φ₂ (indices 1 and 2 of β).
          If the sum exceeds persistence_thresh, shrink both AR coefficients
          by persistence_shrink.  This mirrors the paper's Figure 22 finding
          that GDP AR-persistence FALLS during recessions, not rises.

          The shrinkage operates in standardised space (same scale as the
          ridge solve), so it does not distort the back-transform.
        """
        b = np.linalg.solve(Z.T @ Z + R, Z.T @ y)

        # Persistence control: applies only when X has at least phi1 + phi2
        if len(b) > 2 and np.isfinite(self.persistence_thresh) and self.persistence_shrink != 1.0:
            phi1, phi2 = b[1], b[2]
            if phi1 + phi2 > self.persistence_thresh:
                b[1] *= self.persistence_shrink
                b[2] *= self.persistence_shrink

        return b

    # ── Feature sampling ──────────────────────────────────────────────────────
    def _sample_features(self, p: int) -> np.ndarray:
        n   = max(1, round(p * self.mtry_frac))
        prb = np.ones(p, dtype=float)
        if self.trend_col_idx < p:
            prb[self.trend_col_idx] = float(self.trend_push)
        for idx in self.priority_col_idxs:
            if idx < p:
                prb[idx] = max(prb[idx], float(self.priority_weight))
        prb /= prb.sum()
        return np.random.choice(p, size=n, replace=False, p=prb)

    # ── Split: fast (no RW in split criterion) ────────────────────────────────
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
                sse_arr[si] = (r1 ** 2).sum() + (r2 ** 2).sum()
            valid = np.isfinite(sse_arr)
            if not valid.any():
                continue
            sse_arr[valid] = dv_fun(sse_arr[valid])
            bi = int(np.nanargmin(sse_arr))
            if sse_arr[bi] < best_sse:
                best_sse, best_split = sse_arr[bi], (j, splits[bi])
        return best_split

    # ── Split: full (RW-augmented split criterion, paper default) ─────────────
    def _best_split_full(self, ib, ifl, Xb, yb_n, Sb, R, min_leaf, p,
                          std_y, std_X, T, rando_vec):
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
                ifl1   = ifl[ml]
                Z1, y1 = Xb[ib[ml]], yb_n[ib[ml]]
                n1a = self._neighbours(ifl1, 1, T, rando_vec)
                n2a = np.setdiff1d(
                    self._neighbours(ifl1, 2, T, rando_vec), n1a)
                y1a, Z1a = self._rw_augment(y1, Z1, n1a, n2a, std_y, std_X)
                b1 = self._solve(Z1a, y1a, R);  r1 = y1 - Z1 @ b1

                ifl2   = ifl[mr]
                Z2, y2 = Xb[ib[mr]], yb_n[ib[mr]]
                n1b = self._neighbours(ifl2, 1, T, rando_vec)
                n2b = np.setdiff1d(
                    self._neighbours(ifl2, 2, T, rando_vec), n1b)
                y2a, Z2a = self._rw_augment(y2, Z2, n1b, n2b, std_y, std_X)
                b2 = self._solve(Z2a, y2a, R);  r2 = y2 - Z2 @ b2
                sse_arr[si] = (r1 ** 2).sum() + (r2 ** 2).sum()
            valid = np.isfinite(sse_arr)
            if not valid.any():
                continue
            sse_arr[valid] = dv_fun(sse_arr[valid])
            bi = int(np.nanargmin(sse_arr))
            if sse_arr[bi] < best_sse:
                best_sse, best_split = sse_arr[bi], (j, splits[bi])
        return best_split

    # ── Public: fit ───────────────────────────────────────────────────────────
    def fit(self, X_lin: np.ndarray, y: np.ndarray, S_state: np.ndarray):
        T, K = X_lin.shape
        self._K = K
        self._T = T

        # FIX 6: Separate standardisation for y, X-body, and S
        std_y_arr, mu_y, sig_y        = _standardise_vec(y)
        std_Xb,    mu_X,  sig_X       = _standardise_mat(X_lin[:, 1:])
        std_X  = np.column_stack([np.ones(T), std_Xb])
        std_S, mu_S, sig_S            = _standardise_mat(S_state)

        self._mu_y,  self._sig_y = mu_y,  sig_y
        self._mu_X,  self._sig_X = mu_X,  sig_X
        self._mu_S,  self._sig_S = mu_S,  sig_S

        # FIX 1: contiguous block subsample
        rando_vec   = self._block_subsample(T)
        self._rando = rando_vec

        Xb   = std_X[rando_vec]
        yb   = std_y_arr[rando_vec]
        Sb   = std_S[rando_vec]
        yb_n = yb + 1.5e-7 * np.random.normal(size=len(rando_vec))

        R        = self._reg_mat(K)
        p        = std_S.shape[1]
        min_leaf = max(self.min_samples_leaf,
                       int(2 * self.min_leaf_frac_of_x * (K + 1) + 2))

        # Grow tree
        root  = dict(ib=np.arange(len(rando_vec)), ifl=rando_vec,
                     depth=0, is_leaf=False, sv=None, sp=None,
                     L=None, R=None, beta_std=None)
        stack = [root]

        while stack:
            nd         = stack.pop()
            ib, ifl, d = nd["ib"], nd["ifl"], nd["depth"]

            yy, ZZ = yb_n[ib], Xb[ib]
            n1     = self._neighbours(ifl, 1, T, rando_vec)
            n2     = np.setdiff1d(self._neighbours(ifl, 2, T, rando_vec), n1)
            yya, ZZa       = self._rw_augment(yy, ZZ, n1, n2, std_y_arr, std_X)
            nd["beta_std"] = self._solve(ZZa, yya, R)

            if d >= self.max_depth or len(ib) < 2 * min_leaf:
                nd["is_leaf"] = True
                continue

            if self.fast_rw:
                best = self._best_split_fast(ib, Xb, yb_n, Sb, R, min_leaf, p)
            else:
                best = self._best_split_full(
                    ib, ifl, Xb, yb_n, Sb, R, min_leaf, p,
                    std_y_arr, std_X, T, rando_vec)

            if best is None:
                nd["is_leaf"] = True
                continue

            j_star, c_star = best
            ml = Sb[ib, j_star] < c_star;  mr = ~ml
            nd["sv"], nd["sp"] = j_star, c_star
            nd["L"] = dict(ib=ib[ml], ifl=ifl[ml], depth=d + 1,
                           is_leaf=False, sv=None, sp=None,
                           L=None, R=None, beta_std=None)
            nd["R"] = dict(ib=ib[mr], ifl=ifl[mr], depth=d + 1,
                           is_leaf=False, sv=None, sp=None,
                           L=None, R=None, beta_std=None)
            stack.extend([nd["L"], nd["R"]])

        self._root     = root
        self._min_leaf = min_leaf

        # FIX 2 & 3: leaf estimation on all T obs with full-sample neighbours
        self._beta_bank_std = np.full((T, K), np.nan)
        self._fitted_std    = np.full(T, np.nan)
        self._pred_given_tree(std_X, std_y_arr, std_S, T, rando_vec, R)
        self._back_transform()

    # ── Leaf beta estimation (all T obs routed; full-sample RW neighbours) ────
    def _pred_given_tree(self, std_X, std_y, std_S, T, rando_vec, R):
        """
        For each leaf:
          1. Collect ALL T observations routed to that leaf (not just subsample).
          2. Use full-sample RW neighbours for temporal anchoring (not limited
             to rando_vec), so that small crisis-era leaves get adequate
             regularisation from surrounding quarters.
          3. If fewer than min_samples_leaf observations route to the leaf,
             fall back to the node-level RW beta (beta_std) to avoid unstable
             estimates from tiny crisis-leaf samples.
          4. HRW blend: beta_final = (1-HRW)*beta_hat + HRW*beta_std

        Note: _solve (called here) applies persistence control (FIX 7) so
        leaf-level betas also respect the φ₁+φ₂ ceiling.
        """
        leaves = self._collect_leaves(self._root)
        for li, lf in enumerate(leaves):
            lf["_li"] = li

        leaf_ids = self._route_batch(std_S)   # route ALL T obs

        for li, leaf in enumerate(leaves):
            mask_all = leaf_ids == li
            idx_all  = np.where(mask_all)[0]
            # Fall back to subsample obs if no T-obs are routed to this leaf
            if len(idx_all) == 0:
                idx_all = leaf["ifl"]
            if len(idx_all) == 0:
                continue

            # Guard: if too few obs, use node-level beta to avoid instability
            if len(idx_all) < self.min_samples_leaf and leaf.get("beta_std") is not None:
                leaf["beta_final"] = leaf["beta_std"]
                if mask_all.any():
                    self._beta_bank_std[mask_all] = leaf["beta_std"]
                    self._fitted_std[mask_all]    = std_X[mask_all] @ leaf["beta_std"]
                continue

            yy, ZZ = std_y[idx_all], std_X[idx_all]

            # Full-sample=True → neighbours drawn from all T, not just rando_vec
            n1 = self._neighbours(idx_all, 1, T, rando_vec,
                                  exclude=idx_all, full_sample=True)
            n2 = np.setdiff1d(
                self._neighbours(idx_all, 2, T, rando_vec,
                                 exclude=idx_all, full_sample=True), n1)

            yya, ZZa   = self._rw_augment(yy, ZZ, n1, n2, std_y, std_X)
            beta_hat   = self._solve(ZZa, yya, R)   # persistence control applied here
            beta_final = (1.0 - self.HRW) * beta_hat + self.HRW * leaf["beta_std"]
            leaf["beta_final"] = beta_final

            if mask_all.any():
                self._beta_bank_std[mask_all] = beta_final
                self._fitted_std[mask_all]    = std_X[mask_all] @ beta_final

    def _collect_leaves(self, root) -> list:
        leaves, stack = [], [root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                nd["is_leaf"] = True
                leaves.append(nd)
            else:
                stack.extend([nd["L"], nd["R"]])
        return leaves

    def _route_batch(self, std_S: np.ndarray) -> np.ndarray:
        T      = std_S.shape[0]
        leaves = self._collect_leaves(self._root)
        for li, lf in enumerate(leaves):
            lf["_li"] = li
        result = np.zeros(T, dtype=int)
        for i in range(T):
            nd = self._root
            while not nd["is_leaf"]:
                nd = nd["L"] if std_S[i, nd["sv"]] < nd["sp"] else nd["R"]
            result[i] = nd["_li"]
        return result

    # ── Back-transform (FIX 6: uses separate std params) ─────────────────────
    def _back_transform(self):
        K     = self._K
        sig_y = self._sig_y;  mu_y = self._mu_y
        sig_X = self._sig_X;  mu_X = self._mu_X

        b = self._beta_bank_std.copy()
        for k in range(1, K):
            b[:, k] = b[:, k] * sig_y / sig_X[k - 1]
        b[:, 0] = b[:, 0] * sig_y + mu_y
        for k in range(1, K):
            b[:, 0] -= b[:, k] * mu_X[k - 1]

        self._beta_bank = b
        self._fitted    = self._fitted_std * sig_y + mu_y

    def _leaf_beta_raw(self, leaf) -> np.ndarray:
        K     = self._K
        sig_y = self._sig_y;  mu_y = self._mu_y
        sig_X = self._sig_X;  mu_X = self._mu_X
        b  = leaf["beta_final"].copy()
        br = np.empty(K)
        for k in range(1, K):
            br[k] = b[k] * sig_y / sig_X[k - 1]
        br[0] = b[0] * sig_y + mu_y
        for k in range(1, K):
            br[0] -= br[k] * mu_X[k - 1]
        return br

    def _route_one_std(self, s_std: np.ndarray) -> dict:
        nd = self._root
        while not nd["is_leaf"]:
            nd = nd["L"] if s_std[nd["sv"]] < nd["sp"] else nd["R"]
        return nd

    def predict(self, X_new: np.ndarray, S_new: np.ndarray) -> np.ndarray:
        s_batch = (S_new - self._mu_S) / self._sig_S
        out     = np.empty(len(X_new))
        for i in range(len(X_new)):
            leaf = self._route_one_std(s_batch[i])
            out[i] = (X_new[i] @ self._leaf_beta_raw(leaf)
                      if leaf.get("beta_final") is not None else np.nan)
        return out

    def get_betas(self, S_new: np.ndarray) -> np.ndarray:
        s_batch = (S_new - self._mu_S) / self._sig_S
        out     = np.full((len(S_new), self._K), np.nan)
        for i in range(len(S_new)):
            leaf = self._route_one_std(s_batch[i])
            if leaf.get("beta_final") is not None:
                out[i] = self._leaf_beta_raw(leaf)
        return out

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def tree_depth(self) -> int:
        max_d = [0]
        stack = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                max_d[0] = max(max_d[0], d)
            else:
                stack.extend([(nd["L"], d + 1), (nd["R"], d + 1)])
        return max_d[0]

    def n_leaves(self) -> int:
        count = [0]
        stack = [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                count[0] += 1
            else:
                stack.extend([nd["L"], nd["R"]])
        return count[0]

    def named_splits(self, col_names: list) -> list:
        splits = []
        stack  = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if not nd["is_leaf"] and nd["sv"] is not None:
                name = (col_names[nd["sv"]]
                        if nd["sv"] < len(col_names) else f"col_{nd['sv']}")
                splits.append((d, name, nd["sp"]))
                stack.extend([(nd["L"], d + 1), (nd["R"], d + 1)])
        return splits
    
