"""
single_tree.py — MRFTree: one Macroeconomic Random Forest tree.

Paper reference: Goulet Coulombe (2024) "Macroeconomy as a Random Forest"
Algorithm 1 (Appendix A.6):
  Step 1 : Block-Bayesian-Bootstrap subsample (block_size=8 quarterly, rate=0.75)
  Step 2 : Grow tree recursively — at each node, draw mtry features (with Trend Push),
            fit ridge regressions on each candidate split, keep the split that minimises
            the weighted SSE (DV penalty for edge splits).  Stop when parent < Minimal
            Node Size (=10 quarterly) OR MLF constraint would be violated in children.
  Step 3 : OOS prediction: route (X_t, S_t) → leaf → beta_t → X_t @ beta_t
  Step 4 : OOS GTVPs: average leaf betas across all B trees.

Key hyperparameters (paper defaults, quarterly):
  min_samples_leaf   = 10      (Minimal Node Size — paper App A.6)
  mtry_frac          = 1/3     (paper App A.6, robust to 0.1–0.5 in macro)
  min_leaf_frac_of_x = 1.0     (MLF — paper App A.6; with RWR+RL active MLF=1 OK)
  ridge_lambda       = 0.1     (RL — paper App A.6)
  rw_regul           = 0.75    (RWR / zeta — paper eq. 3)
  HRW                = 0.0     (hierarchical RW blending; 0 = pure leaf ridge)
  subsampling_rate   = 0.75    (paper App A.6)
  block_size         = 8       (quarterly; paper App A.4 / Algorithm 1)
  trend_push         > 1       (pushes trend probability above 1/dim(S))
  fast_rw            = True    (skip RW augmentation during split search; GitHub default)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dv_fun(sse, dv_pref: float = 0.15):
    """
    Down-vote extreme (edge) splits to encourage deeper, more balanced trees.
    Paper App A.6 / GitHub DV_fun.  Applied to the VALID split SSEs only.
    seq runs 1..len(sse), penalty is symmetric and U-shaped — minimum at centre.
    """
    n   = len(sse)
    seq = np.arange(1, n + 1, dtype=float)
    dv  = 0.5 * seq ** 2 - seq
    dv  = dv / np.mean(dv)
    dv  = dv - dv.min() + 1.0          # shift so min = 1
    return sse * (dv ** dv_pref)


def standardise(Y: np.ndarray) -> dict:
    """
    Standardise columns of Y (zero mean, unit std, ddof=1).
    Columns with zero variance are left as-is (std set to 1 to avoid div-by-zero).
    Returns dict with keys 'Y', 'mean', 'std'.
    """
    Y   = np.asarray(Y, dtype=float)
    mu  = Y.mean(axis=0)
    sig = Y.std(axis=0, ddof=1)
    sig[sig == 0] = 1.0
    return {"Y": (Y - mu) / sig, "mean": mu, "std": sig}


# ─────────────────────────────────────────────────────────────────────────────
# MRFTree
# ─────────────────────────────────────────────────────────────────────────────

class MRFTree:
    """
    Single tree for Macroeconomic Random Forest (MRF / FA-ARRF).

    The linear part  X_t  (shape T×K) contains [1, y_{t-1}, y_{t-2}, F1_{t-1}, F2_{t-1}].
    The state space  S_t  (shape T×p) contains all macro indicators used for splitting.
    Time-varying parameters beta_t are estimated by ridge regression inside each leaf,
    optionally augmented with Random-Walk (RW) regularisation rows from temporal neighbours.
    """

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(
        self,
        min_samples_leaf:   int   = 10,
        mtry_frac:          float = 1 / 3,
        ridge_lambda:       float = 0.1,
        rw_regul:           float = 0.1,
        HRW:                float = 0.0,
        subsampling_rate:   float = 0.75,
        block_size:         int   = 8,
        min_leaf_frac_of_x: float = 1.0,
        no_rw_trespassing:  bool  = True,
        max_depth:          int   = 100,
        trend_push:         float = 2.0,
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
        self.trend_push         = float(trend_push)
        self.trend_col_idx      = trend_col_idx
        self.fast_rw            = fast_rw
        self.priority_col_idxs  = list(priority_col_idxs) if priority_col_idxs else []
        self.priority_weight    = float(priority_weight)

    # ── Ridge penalty matrix ──────────────────────────────────────────────────
    def _reg_mat(self, K: int) -> np.ndarray:
        """
        Ridge penalty: lambda * I, but intercept (col 0) penalised lightly (×0.01).
        This matches the GitHub implementation and equation (1) of the paper.
        """
        R       = np.eye(K) * self.ridge_lambda
        R[0, 0] *= 0.01
        return R

    # ── Block-Bayesian Bootstrap subsampling ──────────────────────────────────
    def _block_subsample(self, T: int) -> np.ndarray:
        """
        Paper App A.4 / Algorithm 1 step 1:
        Assign each observation to a block, draw exponential block weights,
        retain observations whose block weight exceeds the (1-rate) quantile.
        block_size=8 for quarterly data (2-year blocks).
        """
        n_blocks = max(1, T // self.block_size)
        # Each observation is assigned to a block index (sorted to keep temporal order)
        groups  = np.sort(np.random.choice(n_blocks, size=T, replace=True))
        # Draw exponential weights for each block (+ small floor to avoid zero)
        w_block = np.random.exponential(1.0, size=n_blocks) + 0.1
        w_obs   = w_block[groups]
        thresh  = np.quantile(w_obs, 1.0 - self.subsampling_rate)
        return np.sort(np.where(w_obs > thresh)[0])

    # ── Minimum leaf size (MLF constraint) ────────────────────────────────────
    def _min_leaf(self, K: int) -> int:
        """
        Paper App A.6: MLF ensures params/obs ratio <= 1/MLF in any leaf.
        With K parameters (including intercept) and MLF, need obs >= MLF * K.
        Also enforced to be at least min_samples_leaf (Minimal Node Size = 10).
        """
        mlf_floor = int(np.ceil(self.min_leaf_frac_of_x * K))
        return max(self.min_samples_leaf, mlf_floor)

    # ── mtry feature sampling with Trend Push ────────────────────────────────
    def _sample_features(self, p: int) -> np.ndarray:
        """
        Paper App A.6 / Algorithm 1 step 2:
        Draw mtry = round(p * mtry_frac) features without replacement.
        Trend (trend_col_idx) gets probability multiplied by trend_push.
        Priority economic indicators get probability multiplied by priority_weight.
        All probabilities are uniform (1/p) by default before adjustments.
        """
        n   = max(1, round(p * self.mtry_frac))
        prb = np.ones(p, dtype=float)
        # Trend Push: multiply trend probability (paper App A.6)
        if 0 <= self.trend_col_idx < p and self.trend_push > 1:
            prb[self.trend_col_idx] = self.trend_push
        # Priority leading indicators get extra weight
        for idx in self.priority_col_idxs:
            if 0 <= idx < p:
                prb[idx] = max(prb[idx], self.priority_weight)
        prb /= prb.sum()
        return np.random.choice(p, size=n, replace=False, p=prb)

    # ── Temporal neighbour lookup ─────────────────────────────────────────────
    def _neighbours(
        self, idx: np.ndarray, n_steps: int, T: int,
        rando_vec: np.ndarray, exclude: np.ndarray = None
    ) -> np.ndarray:
        """
        Find temporal neighbours ±n_steps of idx within the subsample (rando_vec).
        Excludes idx itself (or any custom exclude set) and OOS positions.
        no_rw_trespassing=True (always on, per paper) restricts to rando_vec only.
        """
        idx  = np.asarray(idx)
        nbrs = np.unique(np.concatenate([idx + n_steps, idx - n_steps]))
        excl = np.asarray(exclude if exclude is not None else idx)
        nbrs = nbrs[~np.isin(nbrs, excl)]
        nbrs = nbrs[(nbrs >= 0) & (nbrs < T)]
        if self.no_rw_trespassing:
            nbrs = np.intersect1d(nbrs, rando_vec)
        return nbrs

    # ── RW augmentation (row-append) ─────────────────────────────────────────
    def _rw_augment(
        self, yy: np.ndarray, ZZ: np.ndarray,
        n1: np.ndarray, n2: np.ndarray,
        full_y: np.ndarray, full_Z: np.ndarray
    ):
        """
        Paper eq. (3): augment (yy, ZZ) with RW-regularisation rows from neighbours.
        1-step neighbours get weight rw_regul (zeta).
        2-step neighbours get weight rw_regul^2.
        """
        py = [np.atleast_1d(yy)]
        pZ = [np.atleast_2d(ZZ)]
        if len(n1) > 0:
            py.append(self.rw_regul * full_y[n1])
            pZ.append(self.rw_regul * full_Z[n1])
        if len(n2) > 0:
            py.append(self.rw_regul ** 2 * full_y[n2])
            pZ.append(self.rw_regul ** 2 * full_Z[n2])
        return np.concatenate(py), np.vstack(pZ)

    # ── Ridge solve ───────────────────────────────────────────────────────────
    @staticmethod
    def _solve(Z: np.ndarray, y: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve (Z'Z + R) beta = Z'y via np.linalg.solve (paper eq. 1)."""
        return np.linalg.solve(Z.T @ Z + R, Z.T @ y)

    # ── Split search: fast (no RW during search) ──────────────────────────────
    def _best_split_fast(
        self, ib: np.ndarray, Xb: np.ndarray, yb_n: np.ndarray,
        Sb: np.ndarray, R: np.ndarray, min_leaf: int, p: int
    ):
        """
        fast_rw=True path (GitHub default, paper says this is fine).
        Plain ridge (no RW augmentation) during split search for speed.
        SSE = sum of squared residuals on LOCAL leaf obs (before any RW augment).
        DV penalty applied to valid-split SSEs to prefer middle splits.
        """
        feat_idx = self._sample_features(p)
        best_sse, best_split = np.inf, None

        for j in feat_idx:
            s_vals = Sb[ib, j]
            splits = np.unique(s_vals)
            if len(splits) < 2:
                continue

            # Only consider splits that leave at least min_leaf obs on each side
            valid_splits = [sp for sp in splits
                            if (s_vals < sp).sum() >= min_leaf
                            and (s_vals >= sp).sum() >= min_leaf]
            if not valid_splits:
                continue

            sse_arr = np.empty(len(valid_splits))
            for si, sp in enumerate(valid_splits):
                ml = s_vals < sp
                mr = ~ml
                Z1, y1 = Xb[ib[ml]], yb_n[ib[ml]]
                b1     = self._solve(Z1, y1, R)
                r1     = y1 - Z1 @ b1
                Z2, y2 = Xb[ib[mr]], yb_n[ib[mr]]
                b2     = self._solve(Z2, y2, R)
                r2     = y2 - Z2 @ b2
                sse_arr[si] = (r1 ** 2).sum() + (r2 ** 2).sum()

            # DV penalty on the valid subset (middle of valid splits preferred)
            sse_dv = dv_fun(sse_arr)
            bi     = int(np.argmin(sse_dv))
            if sse_dv[bi] < best_sse:
                best_sse   = sse_dv[bi]
                best_split = (j, valid_splits[bi])

        return best_split

    # ── Split search: full (with RW augmentation during search) ──────────────
    def _best_split_full(
        self, ib: np.ndarray, ifl: np.ndarray,
        Xb: np.ndarray, yb_n: np.ndarray, Sb: np.ndarray,
        R: np.ndarray, min_leaf: int, p: int,
        std_y: np.ndarray, std_X: np.ndarray, T: int, rando_vec: np.ndarray
    ):
        """
        fast_rw=False path: includes RW neighbours when scoring each candidate split.
        Slower but more faithful to the paper's equation (3) during tree growth.
        """
        feat_idx = self._sample_features(p)
        best_sse, best_split = np.inf, None

        for j in feat_idx:
            s_vals = Sb[ib, j]
            splits = np.unique(s_vals)
            if len(splits) < 2:
                continue

            valid_splits = [sp for sp in splits
                            if (s_vals < sp).sum() >= min_leaf
                            and (s_vals >= sp).sum() >= min_leaf]
            if not valid_splits:
                continue

            sse_arr = np.empty(len(valid_splits))
            for si, sp in enumerate(valid_splits):
                ml = s_vals < sp
                mr = ~ml
                ifl1, ifl2 = ifl[ml], ifl[mr]

                Z1, y1 = Xb[ib[ml]], yb_n[ib[ml]]
                n1a    = self._neighbours(ifl1, 1, T, rando_vec)
                n2a    = np.setdiff1d(self._neighbours(ifl1, 2, T, rando_vec), n1a)
                y1a, Z1a = self._rw_augment(y1, Z1, n1a, n2a, std_y, std_X)
                b1     = self._solve(Z1a, y1a, R)
                r1     = y1 - Z1 @ b1          # residuals on LOCAL obs only

                Z2, y2 = Xb[ib[mr]], yb_n[ib[mr]]
                n1b    = self._neighbours(ifl2, 1, T, rando_vec)
                n2b    = np.setdiff1d(self._neighbours(ifl2, 2, T, rando_vec), n1b)
                y2a, Z2a = self._rw_augment(y2, Z2, n1b, n2b, std_y, std_X)
                b2     = self._solve(Z2a, y2a, R)
                r2     = y2 - Z2 @ b2

                sse_arr[si] = (r1 ** 2).sum() + (r2 ** 2).sum()

            sse_dv = dv_fun(sse_arr)
            bi     = int(np.argmin(sse_dv))
            if sse_dv[bi] < best_sse:
                best_sse   = sse_dv[bi]
                best_split = (j, valid_splits[bi])

        return best_split

    # ── Public: fit ───────────────────────────────────────────────────────────
    def fit(self, X_lin: np.ndarray, y: np.ndarray, S_state: np.ndarray):
        """
        Fit one MRF tree.

        Parameters
        ----------
        X_lin   : (T, K)  Linear part — columns [1, y_{t-1}, ..., F2_{t-1}]
        y       : (T,)    Target variable (h-step ahead GDP growth)
        S_state : (T, p)  State-space matrix used for splitting (S_t)
        """
        T, K = X_lin.shape

        # ── Standardise ALL variables jointly (paper: ridge requires standardisation)
        all_data = np.column_stack([y, X_lin[:, 1:], S_state])  # exclude intercept col
        std      = standardise(all_data)
        sd       = std["Y"]
        std_y    = sd[:, 0]                                  # standardised y
        std_X    = np.column_stack([np.ones(T), sd[:, 1:K]]) # intercept + std regressors
        std_S    = sd[:, K:]                                  # standardised S_t

        # Store for predict/get_betas
        self._std = std
        self._K   = K
        self._T   = T

        # ── Block-Bayesian Bootstrap subsample (Algorithm 1 step 1)
        rando_vec   = self._block_subsample(T)
        self._rando = rando_vec

        # Subset to bootstrap sample
        Xb   = std_X[rando_vec]           # (n_boot, K)
        yb   = std_y[rando_vec]           # (n_boot,)
        Sb   = std_S[rando_vec]           # (n_boot, p)
        # Tiny noise for numerical stability (avoids exact ties in splits)
        yb_n = yb + 1.5e-7 * np.random.normal(size=len(rando_vec))

        R = self._reg_mat(K)
        p = std_S.shape[1]

        # ── MLF-enforced minimum leaf size (paper App A.6)
        min_leaf       = self._min_leaf(K)
        self._min_leaf_val = min_leaf

        # ── Grow tree depth-first (Algorithm 1 step 2) ───────────────────────
        # Each node stores: ib (indices into rando_vec), ifl (original T-space indices),
        # depth, split variable (sv) + threshold (sp), children L/R, leaf beta (beta_std)
        root  = dict(ib=np.arange(len(rando_vec)), ifl=rando_vec,
                     depth=0, is_leaf=False, sv=None, sp=None,
                     L=None, R=None, beta_std=None, beta_final=None)
        stack = [root]

        while stack:
            nd         = stack.pop()
            ib, ifl, d = nd["ib"], nd["ifl"], nd["depth"]

            # Compute node-level RW-augmented beta (used as HRW prior / fallback)
            yy, ZZ = yb_n[ib], Xb[ib]
            n1 = self._neighbours(ifl, 1, T, rando_vec)
            n2 = np.setdiff1d(self._neighbours(ifl, 2, T, rando_vec), n1)
            yya, ZZa       = self._rw_augment(yy, ZZ, n1, n2, std_y, std_X)
            nd["beta_std"] = self._solve(ZZa, yya, R)

            # Stopping criteria:
            # (a) max depth reached
            # (b) node too small to split and leave min_leaf on each side
            if d >= self.max_depth or len(ib) < 2 * min_leaf:
                nd["is_leaf"] = True
                continue

            # Find best split
            if self.fast_rw:
                best = self._best_split_fast(ib, Xb, yb_n, Sb, R, min_leaf, p)
            else:
                best = self._best_split_full(
                    ib, ifl, Xb, yb_n, Sb, R, min_leaf, p, std_y, std_X, T, rando_vec)

            if best is None:
                nd["is_leaf"] = True
                continue

            j_star, c_star = best
            ml = Sb[ib, j_star] < c_star
            mr = ~ml

            # Safety: both children must meet min_leaf (should be guaranteed, but be safe)
            if ml.sum() < min_leaf or mr.sum() < min_leaf:
                nd["is_leaf"] = True
                continue

            nd["sv"], nd["sp"] = j_star, c_star
            Lnd = dict(ib=ib[ml], ifl=ifl[ml], depth=d + 1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, beta_std=None, beta_final=None)
            Rnd = dict(ib=ib[mr], ifl=ifl[mr], depth=d + 1, is_leaf=False,
                       sv=None, sp=None, L=None, R=None, beta_std=None, beta_final=None)
            nd["L"], nd["R"] = Lnd, Rnd
            stack.extend([Lnd, Rnd])

        self._root = root

        # ── pred_given_tree: assign final betas to all T rows (Algorithm 1 step 4)
        self._beta_bank_std = np.full((T, K), np.nan)
        self._fitted_std    = np.full(T, np.nan)
        self._pred_given_tree(std_X, std_y, std_S, T, rando_vec, R)
        self._back_transform()

    # ── pred_given_tree: leaf beta estimation + assignment to all T obs ───────
    def _pred_given_tree(
        self, std_X: np.ndarray, std_y: np.ndarray, std_S: np.ndarray,
        T: int, rando_vec: np.ndarray, R: np.ndarray
    ):
        """
        For each leaf:
          1. Fit RW-augmented ridge using the bootstrap obs that landed in the leaf.
          2. Blend with node-level beta via HRW: beta_final = (1-HRW)*beta_hat + HRW*beta_std
          3. Assign beta_final to ALL T observations that route to this leaf.
        This implements Algorithm 1 steps 3 & 4.
        """
        # Collect all leaves
        leaves = []
        stack  = [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                nd["is_leaf"] = True
                leaves.append(nd)
            else:
                stack.extend([nd["L"], nd["R"]])

        # Number leaves for batch routing
        for li, lf in enumerate(leaves):
            lf["_li"] = li

        # Route all T observations (including OOS — needed for OOS prediction)
        leaf_ids = self._route_batch(std_S)   # (T,) integer leaf indices

        for li, leaf in enumerate(leaves):
            ifl = leaf["ifl"]   # bootstrap obs in this leaf (original T-space indices)
            if len(ifl) == 0:
                continue

            # Fit leaf beta with RW augmentation, using ONLY bootstrap obs in this leaf
            # Exclude the leaf's own obs from the neighbour lookup (prediction mode)
            yy, ZZ = std_y[ifl], std_X[ifl]
            n1     = self._neighbours(ifl, 1, T, rando_vec, exclude=ifl)
            n2     = np.setdiff1d(
                         self._neighbours(ifl, 2, T, rando_vec, exclude=ifl), n1)
            yya, ZZa   = self._rw_augment(yy, ZZ, n1, n2, std_y, std_X)
            beta_hat   = self._solve(ZZa, yya, R)

            # HRW blend: pure leaf ridge when HRW=0 (paper default)
            beta_final = (1.0 - self.HRW) * beta_hat + self.HRW * leaf["beta_std"]
            leaf["beta_final"] = beta_final

            # Assign to all T obs routed to this leaf
            mask = leaf_ids == li
            if mask.any():
                self._beta_bank_std[mask] = beta_final
                self._fitted_std[mask]    = std_X[mask] @ beta_final

    # ── Batch routing of T observations through the tree ─────────────────────
    def _route_batch(self, std_S: np.ndarray) -> np.ndarray:
        """
        Route each of the T rows of std_S to its leaf node.
        Returns integer array of leaf indices (into the leaves list).
        """
        T = std_S.shape[0]
        # Re-collect leaves with consistent ordering
        leaves = []
        stack  = [self._root]
        while stack:
            nd = stack.pop()
            if nd["is_leaf"] or nd["L"] is None:
                nd["is_leaf"] = True
                leaves.append(nd)
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

    # ── Back-transform standardised betas to original units ──────────────────
    def _back_transform(self):
        """
        Reverse standardisation to express betas in original (non-standardised) units.
        Matches GitHub back-transform logic exactly.
        """
        K    = self._K
        std  = self._std
        sig_y = std["std"][0];  mu_y  = std["mean"][0]
        sig_X = std["std"][1:K]; mu_X = std["mean"][1:K]

        b = self._beta_bank_std.copy()
        for k in range(1, K):
            b[:, k] = b[:, k] * sig_y / sig_X[k - 1]
        b[:, 0] = b[:, 0] * sig_y + mu_y
        for k in range(1, K):
            b[:, 0] -= b[:, k] * mu_X[k - 1]

        self._beta_bank = b
        self._fitted    = self._fitted_std * sig_y + mu_y

    def _leaf_beta_raw(self, leaf: dict) -> np.ndarray:
        """Back-transform a single leaf's standardised beta_final to original units."""
        K    = self._K
        std  = self._std
        sig_y = std["std"][0];  mu_y  = std["mean"][0]
        sig_X = std["std"][1:K]; mu_X = std["mean"][1:K]

        b  = leaf["beta_final"].copy()
        br = np.empty(K)
        for k in range(1, K):
            br[k] = b[k] * sig_y / sig_X[k - 1]
        br[0] = b[0] * sig_y + mu_y
        for k in range(1, K):
            br[0] -= br[k] * mu_X[k - 1]
        return br

    # ── Route a single standardised S row to its leaf ─────────────────────────
    def _route_one_std(self, s_std: np.ndarray) -> dict:
        nd = self._root
        while not nd["is_leaf"]:
            nd = nd["L"] if s_std[nd["sv"]] < nd["sp"] else nd["R"]
        return nd

    # ── Public: OOS prediction (Algorithm 1 step 3) ───────────────────────────
    def predict(self, X_new: np.ndarray, S_new: np.ndarray) -> np.ndarray:
        """
        Predict for new observations.
        X_new : (n, K) — linear part in ORIGINAL units
        S_new : (n, p) — state space in ORIGINAL units
        Returns (n,) predictions in original units.
        """
        K      = self._K
        std    = self._std
        # Standardise S_new using TRAINING mean/std
        s_batch = (S_new - std["mean"][K:]) / std["std"][K:]
        out     = np.empty(len(X_new))
        for i in range(len(X_new)):
            leaf = self._route_one_std(s_batch[i])
            if leaf.get("beta_final") is None:
                out[i] = np.nan
            else:
                out[i] = X_new[i] @ self._leaf_beta_raw(leaf)
        return out

    # ── Public: OOS beta retrieval (Algorithm 1 step 4) ──────────────────────
    def get_betas(self, S_new: np.ndarray) -> np.ndarray:
        """
        Return time-varying parameters beta_t for new observations.
        S_new : (n, p) — state space in ORIGINAL units
        Returns (n, K) betas in original units.
        """
        K      = self._K
        std    = self._std
        s_batch = (S_new - std["mean"][K:]) / std["std"][K:]
        out     = np.full((len(S_new), K), np.nan)
        for i in range(len(S_new)):
            leaf = self._route_one_std(s_batch[i])
            if leaf.get("beta_final") is not None:
                out[i] = self._leaf_beta_raw(leaf)
        return out

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def tree_depth(self) -> int:
        """Maximum depth of any leaf."""
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
        """Total number of leaf nodes."""
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
        """Return list of (depth, feature_name, threshold) for all internal nodes."""
        splits = []
        stack  = [(self._root, 0)]
        while stack:
            nd, d = stack.pop()
            if not nd["is_leaf"] and nd["sv"] is not None:
                name = (col_names[nd["sv"]]
                        if nd["sv"] < len(col_names)
                        else f"col_{nd['sv']}")
                splits.append((d, name, nd["sp"]))
                stack.extend([(nd["L"], d + 1), (nd["R"], d + 1)])
        return splits
    