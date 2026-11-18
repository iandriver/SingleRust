"""Independent sparse oracle for interleaved groups, bounded by one group block."""

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def verify(path, dest, n, names):
    with h5py.File(path) as src, h5py.File(dest) as out:
        x = ad.io.sparse_dataset(src["X"])
        groups = ad.io.read_elem(out["obs"])
        np.testing.assert_array_equal(groups.index, names)
        np.testing.assert_array_equal(
            ad.io.read_elem(src["var"]).index, ad.io.read_elem(out["var"]).index
        )
        scan = "sum" in out["layers"]

        def matrix(node):
            return ad.io.sparse_dataset(node) if isinstance(node, h5py.Group) else node

        sums = matrix(out["layers/sum"] if scan else out["X"])
        props = matrix(
            out["layers/count_nonzero"] if scan else out["layers/psbulk_props"]
        )
        cells, genes = x.shape
        width = min(n, max(1, 2_000_000 // genes))
        for first in range(0, n, width):
            stop = min(n, first + width)
            shape = (stop - first, genes)
            expected = sparse.csr_matrix(shape, dtype=np.float64)
            hits = sparse.csr_matrix(shape, dtype=np.int32)
            sizes = np.zeros(shape[0], dtype=int)
            for pos in range(first, cells, n):
                block = x[pos : min(cells, pos + shape[0])].astype(np.float64)
                sizes[: block.shape[0]] += 1
                block.resize(shape)
                expected += block
                hits += block != 0
            observed = sparse.csr_matrix(sums[first:stop])
            assert (expected != observed).nnz == 0
            rates = hits if scan else hits.multiply(1 / sizes[:, None])
            difference = (sparse.csr_matrix(props[first:stop]) - rates).tocsr()
            assert not difference.nnz or abs(difference.data).max() < 1e-7
            if not scan:
                np.testing.assert_array_equal(
                    groups.psbulk_cells.iloc[first:stop], sizes
                )
                np.testing.assert_array_equal(
                    groups.psbulk_counts.iloc[first:stop],
                    np.asarray(expected.sum(axis=1)).ravel(),
                )
    print("parity passed", flush=True)
