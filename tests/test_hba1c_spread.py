"""Phase 3f: the subject-cluster SD bootstrap used for the HbA1c spread figure."""

import numpy as np
import pandas as pd

from faceai.report03f import cluster_sd
from faceai.robustness import N_BOOT


def test_cluster_sd_matches_brute_force_resampling():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(30), rng.integers(1, 4, size=30))
    x = 5.5 + rng.normal(0, 0.7, size=len(subjects))
    out = cluster_sd(x, subjects, seed=7)
    assert out["sd"] == np.std(x, ddof=1)

    codes, uniq = pd.factorize(subjects)
    draws = np.random.default_rng(7).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boot = [
        np.std(np.concatenate([x[codes == c] for c in row]), ddof=1) for row in draws
    ]
    np.testing.assert_allclose(out["ci"], np.percentile(boot, [2.5, 97.5]), rtol=1e-9)
    assert out["ci"][0] < out["sd"] < out["ci"][1]
