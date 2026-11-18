"""Edge cases for the count-only pilot: missing labels, empty rows, stored zeros."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from partitioned import aggregate, aggregate_by
from scipy import sparse

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    # An explicit zero in cell 0, an all-zero cell 3, skipped cell 1.
    x = sparse.csr_matrix(
        ([3.0, 0.0, 7.0, 2.0, 4.0], [0, 1, 2, 0, 1], [0, 2, 3, 4, 4, 5]), shape=(5, 4)
    )
    ad.AnnData(x).write_h5ad(p / "in.h5ad")
    labels = np.array([2, -1, 2, 0, 0])
    for workers, chunk in [(1, 1), (4, 2), (4, 7)]:
        aggregate(
            p / "in.h5ad", p / "out.h5ad", labels, ["a", "b", "c"], workers, chunk
        )
        b = ad.read_h5ad(p / "out.h5ad")
        np.testing.assert_array_equal(b.obs_names, ["a", "b", "c"])
        np.testing.assert_array_equal(
            b.X.toarray(), [[0, 4, 0, 0], [0, 0, 0, 0], [5, 0, 0, 0]]
        )
        np.testing.assert_array_equal(
            b.layers["psbulk_props"].toarray(),
            [[0, 0.5, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
        )
        np.testing.assert_array_equal(b.obs.psbulk_cells, [2, 0, 2])
        np.testing.assert_array_equal(b.obs.psbulk_counts, [4, 0, 5])
print("edge cases passed")

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    dest = p / "out.h5ad"
    dest.write_bytes(b"prior output")
    for val in [-1.0, 0.25]:
        ad.AnnData(sparse.csr_matrix([[val]])).write_h5ad(p / "input.h5ad")
        try:
            aggregate(p / "input.h5ad", dest, np.array([0]), ["a"], 1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
        assert dest.read_bytes() == b"prior output"
print("input rejection and output preservation passed")


with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    # Duplicate cell-gene coordinates must count as one detection.
    x = sparse.csr_matrix(([1, 2, 3, 5], [0, 0, 1, 0], [0, 2, 3, 4]), shape=(3, 2))
    obs = pd.DataFrame(
        {"donor": ["d", "d", "e"], "type": ["t", "t", "u"]}, index=["a", "b", "c"]
    )
    ad.AnnData(x, obs=obs).write_h5ad(p / "in.h5ad")
    for mode in ("sum", "mean"):
        aggregate_by(
            p / "in.h5ad",
            p / "out.h5ad",
            "donor",
            "type",
            workers=4,
            chunk=1,
            mode=mode,
        )
        b = ad.read_h5ad(p / "out.h5ad")
        np.testing.assert_array_equal(b.obs_names, ["d_t", "e_t", "d_u", "e_u"])
        np.testing.assert_array_equal(b.obs.donor, ["d", "e", "d", "e"])
        np.testing.assert_array_equal(b.obs.type, ["t", "t", "u", "u"])
        expected = 3 if mode == "sum" else 1.5
        np.testing.assert_array_equal(
            b.X.toarray(), [[expected, expected], [0, 0], [0, 0], [5, 0]]
        )
        np.testing.assert_array_equal(
            b.layers["psbulk_props"].toarray(), [[0.5, 0.5], [0, 0], [0, 0], [1, 0]]
        )
print("named groups, duplicate CSR coordinates and mean passed")
