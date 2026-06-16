//! # `sr_ooc` — out-of-core preprocessing CLI
//!
//! Thin command-line front end over SingleRust's disk-backed streaming ops, so a Python shim
//! (or a shell) can run them on a `.h5ad` without loading it into memory.
//!
//! ```text
//! sr_ooc qc <file.h5ad> [--chunk N]
//! sr_ooc normalize_total <file.h5ad> [--target-sum 1e4] [--log1p] [--chunk N] [--out OUT]
//! sr_ooc log1p <file.h5ad> [--chunk N] [--out OUT]
//! ```
//!
//! `qc` always writes obs/var back into the file in place. `normalize_total`/`log1p` rewrite `X`;
//! with no `--out` they edit the file in place (write a sibling temp, then atomically rename).

use std::path::{Path, PathBuf};

use single_rust::backed::processing::hvg::highly_variable_genes_backed;
use single_rust::backed::processing::pca::pca_backed;
use single_rust::backed::processing::pipeline::preprocess_backed;
use single_rust::backed::processing::qc::qc_metrics_backed;
use single_rust::backed::processing::transformation::{log1p_backed, normalize_total_backed};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage:\n  sr_ooc qc <file.h5ad> [--chunk N]\n  \
             sr_ooc normalize_total <file.h5ad> [--target-sum F] [--log1p] [--out OUT] [--chunk N]\n  \
             sr_ooc log1p <file.h5ad> [--out OUT] [--chunk N]\n  \
             sr_ooc preprocess <file.h5ad> [--target-sum F] [--no-log1p] [--out OUT] [--chunk N]\n  \
             sr_ooc hvg <file.h5ad> [--n-top-genes N] [--chunk N]\n  \
             sr_ooc pca <file.h5ad> [--n-comps N] [--chunk N]   (needs hvg first)"
        );
        std::process::exit(2);
    }
    let cmd = args[1].as_str();
    let input = PathBuf::from(&args[2]);
    let flags = &args[3..];

    let chunk = flag_val(flags, "--chunk").map(|s| s.parse()).transpose()?;
    let out = flag_val(flags, "--out").map(PathBuf::from);

    match cmd {
        "qc" => {
            qc_metrics_backed(&input, chunk)?;
            println!("qc -> {} (in place)", input.display());
        }
        "hvg" => {
            let n_top = flag_val(flags, "--n-top-genes")
                .map(|s| s.parse())
                .transpose()?;
            highly_variable_genes_backed(&input, n_top, chunk)?;
            println!("hvg -> {} (in place)", input.display());
        }
        "pca" => {
            let n_comps = flag_val(flags, "--n-comps")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(50);
            pca_backed(&input, n_comps, None, chunk)?;
            println!("pca -> {} obsm[\"X_pca\"] (in place)", input.display());
        }
        "log1p" => {
            in_place_or_out(&input, out, |inp, outp| log1p_backed(inp, outp, chunk))?;
        }
        "normalize_total" => {
            let target = flag_val(flags, "--target-sum")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(1e4);
            let log1p = flags.iter().any(|f| f == "--log1p");
            in_place_or_out(&input, out, |inp, outp| {
                normalize_total_backed(inp, outp, target, log1p, chunk)
            })?;
        }
        "preprocess" => {
            // Fused QC + normalize_total + log1p (log1p on by default; --no-log1p to disable).
            let target = flag_val(flags, "--target-sum")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(1e4);
            let log1p = !flags.iter().any(|f| f == "--no-log1p");
            in_place_or_out(&input, out, |inp, outp| {
                preprocess_backed(inp, outp, target, log1p, chunk)
            })?;
        }
        other => anyhow::bail!("unknown command '{other}'"),
    }
    Ok(())
}

/// Run a transform that needs an output path. If `out` is `None`, edit `input` in place by
/// writing to a sibling temp file and renaming over the original on success.
fn in_place_or_out(
    input: &Path,
    out: Option<PathBuf>,
    run: impl FnOnce(&Path, &Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    match out {
        Some(outp) => {
            run(input, &outp)?;
            println!("-> {}", outp.display());
        }
        None => {
            let tmp = input.with_extension("h5ad.tmp");
            run(input, &tmp)?;
            std::fs::rename(&tmp, input)?;
            println!("-> {} (in place)", input.display());
        }
    }
    Ok(())
}

/// Value following `name` in the flag list (e.g. `--chunk 1000` -> `Some("1000")`).
fn flag_val<'a>(flags: &'a [String], name: &str) -> Option<&'a str> {
    flags.iter().position(|f| f == name).and_then(|i| flags.get(i + 1)).map(|s| s.as_str())
}
