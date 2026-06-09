use std::ops::{Deref, DerefMut};

use crate::shared::{statistics::ComputeSum, Precision};
use anndata_memory::IMArrayElement;
use anyhow::bail;
use ndarray::{ArrayD, Axis, Ix2};
use single_algebra::Log1P;
use single_algebra::Normalize;
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::Direction;

/// Dispatch an in-place operation over the supported float storage formats of an
/// [`anndata::ArrayData`].
///
/// The three operation arms receive, respectively, the dense `ArrayD`, the CSR matrix,
/// and the CSC matrix payload (each as `f32` or `f64`). Every unsupported container or
/// dtype bails with a uniform message. This collapses the otherwise-identical 12-variant
/// match arms that `log1p` and `normalize_with_type` would each need for all three
/// containers.
macro_rules! dispatch_float_matrix {
    (
        $data:expr,
        |$arr:ident| $array_op:expr,
        |$csr:ident| $csr_op:expr,
        |$csc:ident| $csc_op:expr $(,)?
    ) => {{
        match $data {
            anndata::ArrayData::Array(dyn_array) => match dyn_array {
                anndata::data::DynArray::F32($arr) => $array_op,
                anndata::data::DynArray::F64($arr) => $array_op,
                _ => bail!("dense array: only f32/f64 are supported for this operation"),
            },
            anndata::ArrayData::CsrMatrix(dyn_csr) => match dyn_csr {
                anndata::data::DynCsrMatrix::F32($csr) => $csr_op,
                anndata::data::DynCsrMatrix::F64($csr) => $csr_op,
                _ => bail!("CSR matrix: only f32/f64 are supported for this operation"),
            },
            anndata::ArrayData::CscMatrix(dyn_csc) => match dyn_csc {
                anndata::data::DynCscMatrix::F32($csc) => $csc_op,
                anndata::data::DynCscMatrix::F64($csc) => $csc_op,
                _ => bail!("CSC matrix: only f32/f64 are supported for this operation"),
            },
            anndata::ArrayData::CsrNonCanonical(_) => {
                bail!("CsrNonCanonical matrices are not supported; canonicalize the matrix first.")
            }
            anndata::ArrayData::DataFrame(_) => {
                bail!("DataFrame-backed X is not supported for this operation.")
            }
        }
    }};
}

pub fn normalize_expression(
    matrix: &IMArrayElement,
    expression_target: u32,
    direction: &Direction,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let precision = precision.unwrap_or_default();

    crate::memory::utils::convert_to_float_if_non_float_type(matrix, Some(precision))?;
    // guarantees that the data is in f32 or f64 format!

    match precision {
        Precision::Single => normalize_with_type::<f32>(matrix, expression_target, direction),
        Precision::Double => normalize_with_type::<f64>(matrix, expression_target, direction),
    }
}

pub fn log1p_expression(
    matrix: &IMArrayElement,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let precision = precision.unwrap_or_default();
    crate::memory::utils::convert_to_float_if_non_float_type(matrix, Some(precision))?;
    log1p(matrix)
}

fn log1p(matrix: &IMArrayElement) -> anyhow::Result<()> {
    let mut write_guard = matrix.0.write_inner();
    let data = write_guard.deref_mut();
    dispatch_float_matrix!(
        data,
        |arr| log1p_dense_array(arr),
        |csr| csr.log1p_normalize(),
        |csc| csc.log1p_normalize(),
    )
}

fn normalize_with_type<T>(
    matrix: &IMArrayElement,
    expression_target: u32,
    direction: &Direction,
) -> anyhow::Result<()>
where
    T: FloatOpsTS,
{
    let target = T::from(expression_target).unwrap();

    // Sparse matrices need their row/column sums pre-computed before we take the
    // write guard, because `sum_whole` read-locks the same matrix (locking it while
    // holding the write guard would deadlock). Dense arrays instead compute their
    // sums directly from the array under the write guard.
    let is_dense = {
        let read_guard = matrix.0.read_inner();
        matches!(read_guard.deref(), anndata::ArrayData::Array(_))
    };
    let sums: Vec<T> = if is_dense {
        Vec::new()
    } else {
        matrix.sum_whole(direction)?
    };

    let mut write_guard = matrix.0.write_inner();

    let data = write_guard.deref_mut();

    dispatch_float_matrix!(
        data,
        |arr| normalize_dense_array(arr, target, direction),
        |csr| csr.normalize::<T>(sums.as_slice(), target, direction),
        |csc| csc.normalize::<T>(sums.as_slice(), target, direction),
    )
}

/// Apply `log1p` (natural `ln(1 + x)`) elementwise to a dense array, in place.
///
/// Mirrors the sparse [`Log1P`] implementation so results are identical regardless of
/// whether `X` is stored densely or as a CSR/CSC matrix. Zeros map to `ln(1) = 0`.
fn log1p_dense_array<S>(arr: &mut ArrayD<S>) -> anyhow::Result<()>
where
    S: FloatOpsTS,
{
    arr.mapv_inplace(|v| (S::one() + v).ln());
    Ok(())
}

/// Scale a dense 2D array in place so each row (or column) totals `target`.
///
/// Per-line sums are computed directly from the array in the compute type `U` (the
/// dense analogue of `sum_whole`), while values are stored in type `S`. For
/// `Direction::ROW` each row is scaled to `target`; for `Direction::COLUMN` each column
/// is. Lines whose sum is non-positive are left untouched, mirroring the sparse
/// [`Normalize`] implementation (`scale = target / sum`).
fn normalize_dense_array<S, U>(
    arr: &mut ArrayD<S>,
    target: U,
    direction: &Direction,
) -> anyhow::Result<()>
where
    S: FloatOpsTS,
    U: FloatOpsTS,
{
    let mut view = arr
        .view_mut()
        .into_dimensionality::<Ix2>()
        .map_err(|e| anyhow::anyhow!("Expected a 2D expression matrix: {e}"))?;

    let to_u = |val: S| -> anyhow::Result<U> {
        U::from(val).ok_or_else(|| anyhow::anyhow!("Failed to convert value for scaling"))
    };

    // Scale a single line (row or column) to `target`, skipping non-positive sums.
    let scale_line = |line: &mut ndarray::ArrayViewMut1<S>| -> anyhow::Result<()> {
        let mut sum = U::zero();
        for &val in line.iter() {
            sum += to_u(val)?;
        }
        if sum > U::zero() {
            let scale = target / sum;
            for val in line.iter_mut() {
                *val = S::from(to_u(*val)? * scale)
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert scaled value"))?;
            }
        }
        Ok(())
    };

    match direction {
        Direction::ROW => {
            for mut row in view.outer_iter_mut() {
                scale_line(&mut row)?;
            }
        }
        Direction::COLUMN => {
            for mut col in view.axis_iter_mut(Axis(1)) {
                scale_line(&mut col)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::DynArray;
    use anndata::ArrayData;
    use anndata_memory::IMArrayElement;
    use ndarray::Array2;

    fn dense_element(values: Array2<f64>) -> IMArrayElement {
        let arr: ArrayData = DynArray::from(values).into();
        IMArrayElement::new(arr)
    }

    fn to_dense_f64(elem: &IMArrayElement) -> Array2<f64> {
        crate::shared::convert_to_array_f64(&elem.get_data().unwrap()).unwrap()
    }

    /// log1p on a dense F64 matrix must apply ln(1+x) and no longer panic.
    #[test]
    fn log1p_dense_matches_formula() -> anyhow::Result<()> {
        let elem = dense_element(Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 3.0, 7.0])?);
        log1p_expression(&elem, Some(Precision::Double))?;
        let out = to_dense_f64(&elem);
        let expected = [0.0_f64, 2.0_f64.ln(), 4.0_f64.ln(), 8.0_f64.ln()];
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-9, "got {got}, expected {exp}");
        }
        Ok(())
    }

    /// Row normalization scales each row to the target total; zero rows stay zero.
    #[test]
    fn normalize_dense_rows_sum_to_target() -> anyhow::Result<()> {
        let elem = dense_element(Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 0.0, 0.0])?);
        normalize_expression(&elem, 10, &Direction::ROW, Some(Precision::Double))?;
        let out = to_dense_f64(&elem);
        // Row 0 summed to 4 -> scaled by 10/4: [2.5, 7.5]; row 1 is all-zero -> untouched.
        assert!((out[[0, 0]] - 2.5).abs() < 1e-9);
        assert!((out[[0, 1]] - 7.5).abs() < 1e-9);
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[1, 1]], 0.0);
        Ok(())
    }
}
