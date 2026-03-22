from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import cvxpy as cp


@dataclass
class AlphaModel:
    """
    Modèle d'alpha simple basé sur une régression OLS cross-sectionnelle ou temporelle.

    Idée :
    - fit(X, y) apprend une relation linéaire entre features et rendement cible
    - predict(X) produit un alpha attendu par actif

    Notes :
    - X doit être un DataFrame pandas
    - y doit être une Series pandas alignée sur X.index
    - ce modèle est volontairement simple et sert de squelette propre
    """
    add_intercept: bool = True
    results_: Optional[Any] = field(default=None, init=False)
    feature_names_: Optional[list[str]] = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlphaModel":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X doit être un pandas.DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y doit être un pandas.Series")

        common_index = X.index.intersection(y.index)
        X_fit = X.loc[common_index].copy()
        y_fit = y.loc[common_index].copy()

        if X_fit.empty:
            raise ValueError("Aucune donnée commune entre X et y")

        X_fit = X_fit.replace([np.inf, -np.inf], np.nan).dropna()
        y_fit = y_fit.loc[X_fit.index]

        if y_fit.isna().any():
            valid = ~y_fit.isna()
            X_fit = X_fit.loc[valid]
            y_fit = y_fit.loc[valid]

        X_design = sm.add_constant(X_fit, has_constant="add") if self.add_intercept else X_fit

        self.results_ = sm.OLS(y_fit, X_design).fit()
        self.feature_names_ = list(X_fit.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.results_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant predict()")
        if self.feature_names_ is None:
            raise RuntimeError("Les features apprises sont absentes")

        X_pred = X[self.feature_names_].copy()
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan)

        if X_pred.isna().any().any():
            raise ValueError("X contient des NaN/Inf, nettoyer les données avant predict()")

        X_design = sm.add_constant(X_pred, has_constant="add") if self.add_intercept else X_pred
        preds = self.results_.predict(X_design)
        return pd.Series(preds, index=X.index, name="alpha")


@dataclass
class FactorRiskModel:
    """
    Modèle de risque factoriel :
        r_i,t = c_i + beta_i' f_t + eps_i,t

    Sorties principales :
    - intercepts_ : intercept par actif
    - beta_df_ : expositions factorielles (N x K)
    - residual_var_ : variance spécifique par actif
    - factor_cov_ : covariance des facteurs

    expected_covariance() renvoie :
        Sigma = B Sigma_f B^T + D
    """
    add_intercept: bool = True
    intercepts_: Optional[pd.Series] = field(default=None, init=False)
    beta_df_: Optional[pd.DataFrame] = field(default=None, init=False)
    residual_var_: Optional[pd.Series] = field(default=None, init=False)
    factor_cov_: Optional[pd.DataFrame] = field(default=None, init=False)
    asset_names_: Optional[list[str]] = field(default=None, init=False)
    factor_names_: Optional[list[str]] = field(default=None, init=False)

    def fit(self, returns: pd.DataFrame, factors: pd.DataFrame) -> "FactorRiskModel":
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns doit être un pandas.DataFrame")
        if not isinstance(factors, pd.DataFrame):
            raise TypeError("factors doit être un pandas.DataFrame")

        common_index = returns.index.intersection(factors.index)
        R = returns.loc[common_index].copy()
        F = factors.loc[common_index].copy()

        if R.empty or F.empty:
            raise ValueError("returns et factors doivent partager des dates communes")

        R = R.replace([np.inf, -np.inf], np.nan)
        F = F.replace([np.inf, -np.inf], np.nan)

        valid_rows = ~(R.isna().any(axis=1) | F.isna().any(axis=1))
        R = R.loc[valid_rows]
        F = F.loc[valid_rows]

        if len(R) < 5:
            raise ValueError("Pas assez d'observations pour estimer le modèle factoriel")

        X = sm.add_constant(F, has_constant="add") if self.add_intercept else F

        intercepts: Dict[str, float] = {}
        betas: Dict[str, np.ndarray] = {}
        residual_vars: Dict[str, float] = {}

        for asset in R.columns:
            y = R[asset]
            model = sm.OLS(y, X).fit()

            if self.add_intercept:
                intercepts[asset] = float(model.params.iloc[0])
                betas[asset] = model.params.iloc[1:].to_numpy(dtype=float)
            else:
                intercepts[asset] = 0.0
                betas[asset] = model.params.to_numpy(dtype=float)

            residual_vars[asset] = float(model.resid.var(ddof=X.shape[1]))

        self.intercepts_ = pd.Series(intercepts, name="intercept")
        self.beta_df_ = pd.DataFrame(betas, index=F.columns).T
        self.residual_var_ = pd.Series(residual_vars, name="specific_var")
        self.factor_cov_ = F.cov()
        self.asset_names_ = list(R.columns)
        self.factor_names_ = list(F.columns)

        return self

    def expected_covariance(self) -> pd.DataFrame:
        if self.beta_df_ is None or self.factor_cov_ is None or self.residual_var_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant expected_covariance()")

        B = self.beta_df_.values
        Sigma_f = self.factor_cov_.values
        D = np.diag(self.residual_var_.loc[self.beta_df_.index].values)

        Sigma = B @ Sigma_f @ B.T + D
        Sigma = 0.5 * (Sigma + Sigma.T)

        return pd.DataFrame(Sigma, index=self.beta_df_.index, columns=self.beta_df_.index)

    def exposures(self) -> pd.DataFrame:
        if self.beta_df_ is None:
            raise RuntimeError("Le modèle doit être entraîné avant exposures()")
        return self.beta_df_.copy()


@dataclass
class PortfolioOptimizer:
    """
    Optimiseur de portefeuille convexe avec cvxpy.

    solve(mu, Sigma, exposures, constraints) supporte notamment :
    - sum(w) == net_exposure
    - bornes par actif
    - long_only ou long/short
    - neutralité factorielle
    - pénalité de turnover
    - contrainte de gross exposure

    Exemple de constraints :
    {
        "risk_aversion": 10.0,
        "net_exposure": 1.0,
        "max_weight": 0.05,
        "long_only": False,
        "gross_limit": 1.5,
        "turnover_penalty": 0.001,
        "w_prev": previous_weights,
        "factor_targets": {"MKT": 0.0, "SMB": 0.0}
    }
    """
    solver: str = "OSQP"

    def solve(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        exposures: Optional[pd.DataFrame],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        if constraints is None:
            constraints = {}

        if not isinstance(mu, pd.Series):
            raise TypeError("mu doit être un pandas.Series")
        if not isinstance(Sigma, pd.DataFrame):
            raise TypeError("Sigma doit être un pandas.DataFrame")

        assets = list(mu.index)
        Sigma = Sigma.loc[assets, assets]

        if exposures is not None:
            exposures = exposures.loc[assets]

        risk_aversion = float(constraints.get("risk_aversion", 10.0))
        net_exposure = float(constraints.get("net_exposure", 1.0))
        max_weight = float(constraints.get("max_weight", 0.05))
        long_only = bool(constraints.get("long_only", True))
        gross_limit = constraints.get("gross_limit", None)
        turnover_penalty = float(constraints.get("turnover_penalty", 0.0))
        w_prev = constraints.get("w_prev", None)
        factor_targets = constraints.get("factor_targets", {})

        mu_vec = mu.values.astype(float)
        Sigma_mat = Sigma.values.astype(float)
        Sigma_mat = 0.5 * (Sigma_mat + Sigma_mat.T)

        n_assets = len(assets)
        w = cp.Variable(n_assets)

        objective = mu_vec @ w - risk_aversion * cp.quad_form(w, Sigma_mat)

        if w_prev is not None and turnover_penalty > 0.0:
            if not isinstance(w_prev, pd.Series):
                raise TypeError("w_prev doit être un pandas.Series")
            w_prev_vec = w_prev.loc[assets].values.astype(float)
            objective -= turnover_penalty * cp.norm1(w - w_prev_vec)

        cons = [cp.sum(w) == net_exposure]

        if long_only:
            cons += [w >= 0, w <= max_weight]
        else:
            cons += [cp.abs(w) <= max_weight]

        if gross_limit is not None:
            cons.append(cp.norm1(w) <= float(gross_limit))

        if factor_targets and exposures is None:
            raise ValueError("factor_targets fourni mais exposures est None")

        for factor_name, target in factor_targets.items():
            if factor_name not in exposures.columns:
                raise ValueError(f"Facteur absent des expositions: {factor_name}")
            b = exposures[factor_name].values.astype(float)
            cons.append(b @ w == float(target))

        problem = cp.Problem(cp.Maximize(objective), cons)
        problem.solve(solver=self.solver)

        if w.value is None:
            raise RuntimeError(f"Optimisation échouée. Statut: {problem.status}")

        weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
        return weights


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Exemple minimal exécutable avec données simulées
    # ---------------------------------------------------------------------
    rng = np.random.default_rng(42)

    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    assets = [f"Asset_{i}" for i in range(10)]
    factor_names = ["MKT", "SMB", "HML"]

    # Facteurs simulés
    factors = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(dates), len(factor_names))),
        index=dates,
        columns=factor_names,
    )

    # Vraies expositions simulées
    true_betas = pd.DataFrame(
        rng.normal(0.0, 0.5, size=(len(assets), len(factor_names))),
        index=assets,
        columns=factor_names,
    )

    specific_noise = rng.normal(0.0, 0.02, size=(len(dates), len(assets)))
    returns = pd.DataFrame(
        factors.values @ true_betas.T.values + specific_noise,
        index=dates,
        columns=assets,
    )

    # Features simulées pour modèle d'alpha
    X_alpha = pd.DataFrame(
        rng.normal(size=(len(assets), 4)),
        index=assets,
        columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    )
    y_alpha = pd.Series(
        0.01 * X_alpha["feat_1"] - 0.015 * X_alpha["feat_2"] + rng.normal(0, 0.01, len(assets)),
        index=assets,
        name="target_return",
    )

    # 1) Alpha model
    alpha_model = AlphaModel().fit(X_alpha, y_alpha)
    alpha = alpha_model.predict(X_alpha)

    # 2) Risk model
    risk_model = FactorRiskModel().fit(returns, factors)
    Sigma = risk_model.expected_covariance()
    expo = risk_model.exposures()

    # 3) Optimisation
    optimizer = PortfolioOptimizer(solver="OSQP")
    weights = optimizer.solve(
        mu=alpha,
        Sigma=Sigma,
        exposures=expo,
        constraints={
            "risk_aversion": 5.0,
            "net_exposure": 1.0,
            "max_weight": 0.25,
            "long_only": False,
            "gross_limit": 2.0,
            "factor_targets": {"MKT": 0.0},
        },
    )

    print("=== Alpha prédit ===")
    print(alpha.round(4))
    print("\n=== Expositions factorielles estimées ===")
    print(expo.round(4))
    print("\n=== Poids optimisés ===")
    print(weights.round(4))
    print("\nSomme des poids:", round(weights.sum(), 6))
    print("Gross exposure:", round(np.abs(weights).sum(), 6))
