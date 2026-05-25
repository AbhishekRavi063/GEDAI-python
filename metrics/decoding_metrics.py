"""Motor imagery decoding metrics.

Pipelines:
  'csp_lda'    – CSP → LDA (default, standard BCI baseline)
  'riemannian' – Covariance → Tangent Space → LDA
                 (pyriemann; better on weak/noisy MI signal like Weibo2014)

No data leakage: fitting is always on train set only.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_riemannian_pipeline(random_state: int = 42) -> Pipeline:
    """Riemannian geometry pipeline: Covariance → Tangent Space → LDA.

    Matches MATLAB pyriemann / Brainstorm Riemannian classifier.
    Works directly on covariance matrices on the SPD manifold — no need
    for CSP spatial filtering. Much more robust when MI signal is weak
    (e.g. Weibo2014 where ba_baseline < 0.5 with CSP+LDA).

    Input to pipeline: (n_epochs, n_ch, n_times) raw epochs.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    cov = Covariances(estimator="lwf")   # Ledoit-Wolf shrinkage covariance
    ts  = TangentSpace(metric="riemann") # project SPD matrices to tangent space
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("cov", cov), ("ts", ts), ("lda", lda)])


def build_csp_lda_pipeline(
    n_components: int = 6,
    random_state: int = 42,
    reg: str | float | None = "ledoit_wolf",
) -> Pipeline:
    """CSP + LDA pipeline for motor imagery decoding.

    Defaults updated for robustness across datasets:
      - reg='ledoit_wolf' adds shrinkage to CSP covariance estimation
        (helps with noisy data like Weibo, doesn't hurt clean data)
      - LDA also uses shrinkage solver for robustness
    """
    from mne.decoding import CSP

    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("csp", csp), ("lda", lda)])


def adaptive_n_csp_components(n_channels: int) -> int:
    """Scale CSP component count to channel count.

    For 22 ch → 6 components (standard).
    For 14 ch → 4 (avoid over-parameterization on small montages).
    For 60 ch → 8 (more components for richer spatial info).
    """
    if n_channels < 16:
        return min(4, n_channels // 2)
    if n_channels >= 50:
        return 8
    return 6


def cross_val_balanced_accuracy(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_csp_components: int = 6,
    random_state: int = 42,
    classifier: str = "csp_lda",
) -> dict:
    """Stratified k-fold cross-validated balanced accuracy.

    Parameters
    ----------
    epochs_data : (n_epochs, n_ch, n_times)
    labels : (n_epochs,) int

    Returns
    -------
    dict with 'balanced_accuracy', 'std', 'scores_per_fold', 'kappa', 'ci_95'
    """
    from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

    if classifier == "riemannian":
        pipeline = build_riemannian_pipeline(random_state)
    else:
        # Use adaptive component count if default (6) — scales with channel count
        if n_csp_components == 6 and epochs_data.ndim == 3:
            n_csp_components = adaptive_n_csp_components(epochs_data.shape[1])
        pipeline = build_csp_lda_pipeline(n_csp_components, random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    kappas = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(epochs_data, labels)):
        X_train, X_test = epochs_data[train_idx], epochs_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            ba = balanced_accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            scores.append(ba)
            kappas.append(kappa)
        except Exception as exc:
            logger.warning(f"Fold {fold_idx} failed: {exc}")

    if not scores:
        return {"balanced_accuracy": float("nan"), "std": float("nan"),
                "scores_per_fold": [], "kappa": float("nan"), "ci_95": (float("nan"), float("nan"))}

    scores_arr = np.array(scores)
    kappa_arr = np.array(kappas)
    ci = (float(np.percentile(scores_arr, 2.5)), float(np.percentile(scores_arr, 97.5)))

    return {
        "balanced_accuracy": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr)),
        "scores_per_fold": scores_arr.tolist(),
        "kappa": float(np.mean(kappa_arr)),
        "ci_95": ci,
    }


def permutation_test_accuracy(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Permutation test: p-value that observed accuracy > chance.

    Parameters
    ----------
    n_permutations : int – number of permutation iterations

    Returns
    -------
    dict with 'p_value', 'observed_accuracy', 'chance_distribution'
    """
    rng = np.random.default_rng(random_state)

    obs = cross_val_balanced_accuracy(epochs_data, labels, n_splits, random_state=random_state)
    observed_acc = obs["balanced_accuracy"]

    chance_scores = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        res = cross_val_balanced_accuracy(epochs_data, perm_labels, n_splits=3, random_state=int(rng.integers(1000)))
        chance_scores.append(res["balanced_accuracy"])

    chance_arr = np.array(chance_scores)
    p_value = float(np.mean(chance_arr >= observed_acc))

    return {
        "p_value": p_value,
        "observed_accuracy": observed_acc,
        "chance_mean": float(np.mean(chance_arr)),
        "chance_std": float(np.std(chance_arr)),
        "chance_distribution": chance_arr.tolist(),
    }


def compute_all_decoding_metrics(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    run_permutation: bool = False,
    random_state: int = 42,
    classifier: str = "csp_lda",
) -> dict:
    """All decoding metrics in one call."""
    m = cross_val_balanced_accuracy(
        epochs_data, labels, n_splits,
        random_state=random_state, classifier=classifier,
    )
    if run_permutation:
        m["permutation"] = permutation_test_accuracy(
            epochs_data, labels, n_permutations=100, random_state=random_state
        )
    return m


def train_test_balanced_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    classifier: str = "csp_lda",
) -> dict:
    """Train on X_train, evaluate on X_test — no CV, proper held-out evaluation.

    Parameters
    ----------
    X_train : (n_train, n_ch, n_times)
    X_test  : (n_test,  n_ch, n_times)
    """
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

    if classifier == "riemannian":
        pipeline = build_riemannian_pipeline(random_state)
    else:
        n_csp = adaptive_n_csp_components(X_train.shape[1]) if X_train.ndim == 3 else 6
        pipeline = build_csp_lda_pipeline(n_csp, random_state)

    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        ba = float(balanced_accuracy_score(y_test, y_pred))
        kappa = float(cohen_kappa_score(y_test, y_pred))
    except Exception as exc:
        logger.warning(f"train_test_balanced_accuracy failed: {exc}")
        return {"balanced_accuracy": float("nan"), "std": float("nan"),
                "scores_per_fold": [], "kappa": float("nan"),
                "ci_95": (float("nan"), float("nan"))}

    return {"balanced_accuracy": ba, "std": float("nan"),
            "scores_per_fold": [ba], "kappa": kappa,
            "ci_95": (float("nan"), float("nan"))}
