import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "scripts"))
import pipeline as p  # noqa: E402


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"output": 0.0, "a": rng.normal(size=n), "b": rng.normal(size=n),
                       "kind": rng.choice(["x", "y"], n)})
    df["output"] = 2 * df.a - df.b + (df.kind == "x") + rng.normal(0, .1, n)
    df.loc[3, "a"] = np.nan
    return df


def test_remove_duplicates():
    df = pd.concat([toy(), toy().iloc[:5]])
    assert len(p.remove_duplicates(df)) == 200


def test_split_deterministic_and_disjoint():
    a, b = p.split_train_val(toy()), p.split_train_val(toy())
    assert a[0].index.equals(b[0].index)
    assert not set(a[0].index) & set(a[1].index)
    assert len(a[0]) + len(a[1]) == 200


def test_outlier_bounds_use_train_only():
    b = p.iqr_bounds(pd.DataFrame({"a": np.arange(100.0)}), ["a"])
    assert p.clip_outliers(pd.DataFrame({"a": [1e6]}), b)["a"][0] == b["a"][1]


def test_preprocessing_handles_missing_and_categorical():
    X_tr, _, y_tr, _ = p.split_train_val(toy())
    m = p.build_models(X_tr)["Ridge"].fit(X_tr, y_tr)
    assert np.isfinite(m.predict(X_tr)).all()


def test_end_to_end(tmp_path):
    toy().to_csv(tmp_path / "tr.csv", index=False)
    toy(100, 1).to_csv(tmp_path / "te.csv", index=False)
    t = p.run(tmp_path / "tr.csv", tmp_path / "te.csv", tmp_path / "res")
    assert set(t.model) == {"Ridge", "RandomForest", "SVR"}
    assert (tmp_path / "res/figures/model_comparison.png").exists()
    assert t.set_index("model").loc["Ridge", "test_R2"] > 0.9
