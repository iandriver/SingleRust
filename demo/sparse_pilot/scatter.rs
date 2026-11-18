// Inputs come from validated contiguous NumPy arrays in partitioned.py.
#[no_mangle]
pub unsafe extern "C" fn scatter(
    records: *const u32,
    len: usize,
    sums: *mut f64,
    hits: *mut u32,
    size: usize,
) -> bool {
    let records = std::slice::from_raw_parts(records, len * 2);
    let sums = std::slice::from_raw_parts_mut(sums, size);
    let hits = std::slice::from_raw_parts_mut(hits, size);
    for record in records.chunks_exact(2) {
        let index = record[0] as usize;
        if index >= size {
            return false;
        }
        let count = record[1];
        if count > 0 {
            sums[index] += f64::from(count);
            hits[index] += 1;
        }
    }
    true
}
