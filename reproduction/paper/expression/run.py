#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import importlib.util
import math
import pickle
import random
import zlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import simulate_transcript_parallel
from util import NONSTOP_IDX, build_full_rate

try:
    import RNA
except Exception:
    RNA = None


def load_stasep_params(cell_line, params_json=None):
    """Fitted sTASEP wait times: from a fit output JSON if given, else seq2ribo/constants.py.

    alpha indexes the pair count, beta the angle bin, buckets the CDS third.
    """
    if params_json:
        with open(params_json) as f:
            p = json.load(f)["params"]
        return (np.asarray(p["rates"], dtype=np.float64), np.asarray(p["alpha"], dtype=np.float64),
                np.asarray(p["beta"], dtype=np.float64), np.asarray(p["buckets"], dtype=np.float64))
    spec = importlib.util.spec_from_file_location(
        "seq2ribo_constants", Path(__file__).resolve().parents[3] / "seq2ribo" / "constants.py")
    const = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(const)
    key = {"ipsc": "IPSC", "hek293": "HEK293", "rpe": "RPE", "lcl": "LCL"}[cell_line]
    return tuple(np.asarray(getattr(const, f"{key}_{name}"), dtype=np.float64)
                 for name in ("RATES", "ALPHA", "BETA", "BUCKETS"))


def _seeded_simulation(job):
    seed, args = job
    random.seed(seed)
    return simulate_transcript_parallel(args)


ANGLE_BINS = np.array([0.0, 1.0471990, 2.09439815, 3.14159722, 6.51421177], dtype=np.float64)
VALID_RNA = set("ACGU")
STOP_CODONS = {"UAA", "UAG", "UGA"}
ALLOWED_START_CODONS = {"AUG", "CUG", "GUG"}


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], default="ipsc")
    parser.add_argument("--csv", type=Path, default=base_dir / "expression" / "mRFP_Expression.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--no-fold", action="store_true", help="skip ViennaRNA folding (all-zero structure features)")
    parser.add_argument("--params", default=None,
                        help="fit output JSON (sTASEP); default: seq2ribo/constants.py for --cell-line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace existing samples")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = base_dir / "results" / "expression" / args.cell_line / "run"
    return args


def _normalized_rate(rate_fit: np.ndarray) -> np.ndarray:
    full = build_full_rate(rate_fit)
    mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
    full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
    return full


def _clean_rna(seq: str) -> tuple[str, str | None]:
    s = (seq or "").upper().replace("T", "U").replace(" ", "")
    if set(s) - VALID_RNA:
        s = "".join([c for c in s if c in VALID_RNA])
    if len(s) < 3:
        return "", "too_short"
    if len(s) % 3 != 0:
        s = s[: len(s) - (len(s) % 3)]
        return s, "truncated"
    return s, None


def _starts_with_start(seq: str) -> bool:
    return len(seq) >= 3 and seq[:3] in ALLOWED_START_CODONS


def _ends_with_stop(seq: str) -> bool:
    return len(seq) >= 3 and (len(seq) % 3) == 0 and seq[-3:] in STOP_CODONS


def _has_internal_stop(seq: str) -> bool:
    if len(seq) < 6 or (len(seq) % 3) != 0:
        return False
    for i in range(0, len(seq) - 3, 3):
        if seq[i : i + 3] in STOP_CODONS:
            return True
    return False


def _fold_dot(seq: str, max_len: int = 10000) -> str:
    try:
        if RNA is None or len(seq) > max_len:
            return ""
        fc = RNA.fold_compound(seq)
        if fc is None:
            return ""
        struct, _ = fc.mfe()
        return struct
    except Exception:
        return ""


def _compute_vertex_angles_from_ss(seq: str, ss: str):
    try:
        RNA.cvar.rna_plot_type = RNA.PLOT_TYPE_TURTLE
        coords = RNA.get_xy_coordinates(ss)
        xs, ys = [], []
        for i in range(len(seq)):
            p = coords.get(i)
            xs.append(p.X)
            ys.append(p.Y)
        vertex_angles = []
        for i in range(1, len(xs) - 1):
            v1x, v1y = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            v2x, v2y = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 == 0 or n2 == 0:
                ang = 0.0
            else:
                dot = v1x * v2x + v1y * v2y
                cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
                ang = math.acos(cosang)
            vertex_angles.append(ang)
        vertex_angles = [0.0, 0.0] + vertex_angles
        if len(vertex_angles) < len(seq):
            vertex_angles += [0.0] * (len(seq) - len(vertex_angles))
        elif len(vertex_angles) > len(seq):
            vertex_angles = vertex_angles[: len(seq)]
        return vertex_angles
    except Exception:
        return None


def _compute_geometry_features(seq: str, ss: str):
    try:
        n = len(seq)
        if not ss or len(ss) != n:
            return None
        pt = RNA.ptable(ss)
        is_paired = np.array([1 if pt[i + 1] > 0 else 0 for i in range(n)], dtype=np.int32)
        va = _compute_vertex_angles_from_ss(seq, ss)
        if va is None:
            return None
        va = np.asarray(va, dtype=np.float64)
        angle_dev_sum = np.add.reduceat(va, np.arange(0, n, 3))
        pair_count = np.add.reduceat(is_paired, np.arange(0, n, 3)).astype(np.int32)
        lcod = n // 3
        if angle_dev_sum.shape[0] != lcod or pair_count.shape[0] != lcod:
            return None
        return angle_dev_sum, pair_count
    except Exception:
        return None


def _dot_paths(cache_dir: Path, tx: str) -> tuple[Path, Path]:
    dot_dir = cache_dir / "dot_bracket"
    dot_dir.mkdir(parents=True, exist_ok=True)
    return dot_dir / f"{tx}.dot.pkl", dot_dir / f"{tx}.dbn"


def _geom_path(cache_dir: Path, tx: str) -> Path:
    gdir = cache_dir / "geometry"
    gdir.mkdir(parents=True, exist_ok=True)
    return gdir / f"{tx}.geo.pkl"


def _write_dot_success(dot_pkl: Path, dot_txt: Path, ss: str) -> None:
    dot_pkl.write_bytes(pickle.dumps(ss))
    dot_txt.write_text(ss + "\n")


def _load_dot_with_retry(seq: str, dot_pkl: Path, dot_txt: Path) -> str | None:
    if dot_txt.exists():
        try:
            ss = dot_txt.read_text().strip()
            if len(ss) == len(seq):
                return ss
        except Exception:
            pass
    ss_cached = None
    if dot_pkl.exists():
        try:
            obj = pickle.loads(dot_pkl.read_bytes())
            if isinstance(obj, str):
                ss_cached = obj
        except Exception:
            ss_cached = None
    if ss_cached is None:
        ss = _fold_dot(seq)
        if ss == "":
            dot_pkl.write_bytes(pickle.dumps(""))
            return None
        _write_dot_success(dot_pkl, dot_txt, ss)
        return ss
    if ss_cached == "":
        ss = _fold_dot(seq)
        if ss == "":
            return None
        _write_dot_success(dot_pkl, dot_txt, ss)
        return ss
    if len(ss_cached) != len(seq):
        ss = _fold_dot(seq)
        if ss == "":
            dot_pkl.write_bytes(pickle.dumps(""))
            return None
        _write_dot_success(dot_pkl, dot_txt, ss)
        return ss
    return ss_cached


def _load_geometry_with_retry_from_ss(seq: str, ss: str, gfile: Path):
    gobj = None
    if gfile.exists():
        try:
            gobj = pickle.loads(gfile.read_bytes())
        except Exception:
            gobj = None
    if isinstance(gobj, dict):
        ang = gobj.get("angle_dev_sum")
        pc = gobj.get("pair_count")
        if isinstance(ang, np.ndarray) and isinstance(pc, np.ndarray):
            return ang, pc
    feats = _compute_geometry_features(seq, ss)
    if feats is None:
        pickle.dump({"angle_dev_sum": None, "pair_count": None}, gfile.open("wb"))
        return None
    ang, pc = feats
    pickle.dump({"angle_dev_sum": ang, "pair_count": pc}, gfile.open("wb"))
    return ang, pc


def _read_csv_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"Sequence", "Value", "Dataset", "Split"}
        if not reader.fieldnames or not set(reader.fieldnames) >= needed:
            raise ValueError(f"CSV must have columns: {needed}")
        for i, r in enumerate(reader):
            try:
                val = float(r["Value"])
            except Exception:
                val = float("nan")
            rows.append(
                {
                    "idx": i,
                    "Sequence": r["Sequence"],
                    "Value": val,
                    "Dataset": str(r["Dataset"]),
                    "Split": str(r["Split"]),
                }
            )
    return rows


def _pearson_safe(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    return float((xm * ym).sum() / denom) if denom != 0 else float("nan")


def _scatter_with_fit(x, y, xlabel: str, ylabel: str, title: str, outpath: Path) -> float:
    fig = plt.figure(dpi=150)
    plt.scatter(x, y, alpha=0.7)
    if len(x) >= 2:
        try:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            plt.plot(xs, m * xs + b, linewidth=2)
        except Exception:
            pass
    r = _pearson_safe(x, y)
    if len(x) > 0 and len(y) > 0:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        xpad = 0.03 * max(1e-9, (xmax - xmin))
        ypad = 0.05 * max(1e-9, (ymax - ymin))
        plt.text(xmin + xpad, ymax - ypad, f"n={len(x)}; r={r:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return r


def _zeros_like_seq(seq_u: str):
    lcod = len(seq_u) // 3
    a_cnts = np.zeros(lcod, dtype=np.float64)
    return lcod, a_cnts


def _make_job_from_row(row, cache_dir: Path, use_geometry: bool, rate, alpha_vec, beta_vec, bucket_vec):
    seq_u, clean_flag = _clean_rna(row["Sequence"])
    if not seq_u:
        return None, {"status": "skip", "reason": clean_flag or "empty"}
    if not _starts_with_start(seq_u):
        return None, {"status": "skip", "reason": "no_start"}
    if not _ends_with_stop(seq_u):
        return None, {"status": "skip", "reason": "no_stop"}
    if _has_internal_stop(seq_u):
        return None, {"status": "skip", "reason": "internal_stop"}

    angle_dev_sum = None
    pair_count = None
    if use_geometry and RNA is not None:
        cache_key = f"{row['Dataset']}__{row['Split']}__row{row['idx']:06d}"
        dot_pkl, dot_txt = _dot_paths(cache_dir, cache_key)
        ss = _load_dot_with_retry(seq_u, dot_pkl, dot_txt)
        if ss is not None:
            gfile = _geom_path(cache_dir, cache_key)
            feats = _load_geometry_with_retry_from_ss(seq_u, ss, gfile)
            if feats is not None:
                angle_dev_sum, pair_count = feats

    lcod, a_cnts = _zeros_like_seq(seq_u)
    if angle_dev_sum is None or pair_count is None or len(angle_dev_sum) != lcod or len(pair_count) != lcod:
        angle_dev_sum = np.zeros(lcod, dtype=np.float64)
        pair_count = np.zeros(lcod, dtype=np.int32)
        geom_flag = "geom_zero"
    else:
        geom_flag = "geom_ok"

    tx_id = f"{row['Dataset']}|{row['Split']}|row{row['idx']:06d}"
    job = (
        tx_id,
        seq_u,
        a_cnts,
        angle_dev_sum,
        pair_count,
        rate,
        alpha_vec,
        beta_vec,
        bucket_vec,
        ANGLE_BINS,
    )
    meta = {
        "tx": tx_id,
        "expr": float(row["Value"]),
        "dataset": row["Dataset"],
        "split": row["Split"],
        "len_nt": len(seq_u),
        "status": "ok",
        "clean_flag": clean_flag or "",
        "geom_flag": geom_flag if RNA is not None else "no_fold_lib",
    }
    return job, meta


def _run_jobs(jobs, out_dir_pkls: Path, reps: int, seed: int, workers, overwrite: bool):
    out_dir_pkls.mkdir(parents=True, exist_ok=True)
    stale = list(out_dir_pkls.glob("*_sample_*.pkl"))
    if stale:
        if not overwrite:
            raise SystemExit(f"{out_dir_pkls} already has {len(stale)} samples; pass --overwrite to replace them")
        for f in stale:
            f.unlink()
    results_by_tx = {}
    for rep in range(reps):
        seeded = [(zlib.crc32(f"{seed}:{rep}:{j[0]}".encode()), j) for (j, _) in jobs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            sim_out = list(ex.map(_seeded_simulation, seeded, chunksize=8))
        for ((_, meta), res) in zip(jobs, sim_out):
            if meta["status"] != "ok":
                continue
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = res
            out_pkl = out_dir_pkls / f"{tx}_sample_{rep}.pkl"
            with out_pkl.open("wb") as fh:
                pickle.dump((tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total), fh)
            raw_total = float(np.asarray(sim_vec_raw, dtype=np.float64).sum()) if sim_vec_raw is not None else float("nan")
            scaled_total = (
                float(np.asarray(sim_vec_scaled, dtype=np.float64).sum()) if sim_vec_scaled is not None else float("nan")
            )
            use_raw = (not np.isfinite(scaled_total)) or (scaled_total <= 0.0)
            chosen_total = raw_total if use_raw else scaled_total
            chosen_src = "raw" if use_raw else "scaled"
            comp_total = (
                float(np.asarray(completed_total, dtype=np.float64).mean())
                if np.ndim(completed_total)
                else float(completed_total)
            )
            bucket = results_by_tx.setdefault(
                tx,
                {
                    "pred_totals": [],
                    "pred_sources": [],
                    "raw_totals": [],
                    "scaled_totals": [],
                    "completed_totals": [],
                },
            )
            bucket["pred_totals"].append(chosen_total)
            bucket["pred_sources"].append(chosen_src)
            bucket["raw_totals"].append(raw_total)
            bucket["scaled_totals"].append(scaled_total)
            bucket["completed_totals"].append(comp_total)
    finalized = {}
    for tx, d in results_by_tx.items():
        pt = np.asarray(d["pred_totals"], dtype=np.float64)
        rt = np.asarray(d["raw_totals"], dtype=np.float64)
        st = np.asarray(d["scaled_totals"], dtype=np.float64)
        ct = np.asarray(d["completed_totals"], dtype=np.float64)
        src_majority = "raw" if d["pred_sources"].count("raw") >= d["pred_sources"].count("scaled") else "scaled"
        finalized[tx] = {
            "pred_total_mean": float(np.nanmean(pt)),
            "pred_total_std": float(np.nanstd(pt, ddof=0)),
            "pred_source": src_majority,
            "pred_total_raw_mean": float(np.nanmean(rt)),
            "pred_total_raw_std": float(np.nanstd(rt, ddof=0)),
            "pred_total_scaled_mean": float(np.nanmean(st)),
            "pred_total_scaled_std": float(np.nanstd(st, ddof=0)),
            "completed_total_mean": float(np.nanmean(ct)),
            "completed_total_std": float(np.nanstd(ct, ddof=0)),
            "replicas": int(len(pt)),
        }
    return finalized


def main() -> None:
    args = _parse_args()
    rates, alpha_vec, beta_vec, bucket_vec = load_stasep_params(args.cell_line, args.params)
    rate = _normalized_rate(rates)
    if RNA is None and not args.no_fold:
        raise SystemExit("ViennaRNA (python module RNA) is required for structure features; use --no-fold to skip")

    out_dir = args.out_dir
    pkl_dir = out_dir / "pkls"
    cache_dir = out_dir / "fold_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    jobs, metas = [], []
    use_geometry = (not args.no_fold) and (RNA is not None)
    skipped = 0
    for r in rows:
        job, meta = _make_job_from_row(r, cache_dir, use_geometry, rate, alpha_vec, beta_vec, bucket_vec)
        metas.append(meta)
        if job is None:
            skipped += 1
        else:
            jobs.append((job, meta))
    print(f"Prepared {len(jobs)} jobs ({skipped} skipped). Folding={'on' if use_geometry else 'off/zero-geom'}.")

    aggregated = _run_jobs(jobs, pkl_dir, reps=max(1, args.reps), seed=args.seed, workers=args.workers,
                           overwrite=args.overwrite)
    print(f"Simulated {len(aggregated)} sequences with {args.reps} replicate(s).")

    agg_rows, xs, ys = [], [], []
    for meta in metas:
        tx = meta.get("tx")
        if meta["status"] != "ok" or tx not in aggregated:
            continue
        a = aggregated[tx]
        expr = float(meta["expr"])
        y = a["pred_total_mean"]
        if not (math.isnan(expr) or math.isnan(y) or math.isinf(expr) or math.isinf(y)):
            xs.append(expr)
            ys.append(y)
        agg_rows.append(
            {
                "tx": tx,
                "dataset": meta["dataset"],
                "split": meta["split"],
                "len_nt": meta["len_nt"],
                "expression_value": expr,
                "pred_total_mean": a["pred_total_mean"],
                "pred_total_std": a["pred_total_std"],
                "completed_total_mean": a["completed_total_mean"],
                "completed_total_std": a["completed_total_std"],
                "replicas": a["replicas"],
                "clean_flag": meta["clean_flag"],
                "geom_flag": meta["geom_flag"],
            }
        )

    agg_csv = out_dir / "stasep_predicted_totals_vs_expression.csv"
    with agg_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "tx",
                "dataset",
                "split",
                "len_nt",
                "expression_value",
                "pred_total_mean",
                "pred_total_std",
                "completed_total_mean",
                "completed_total_std",
                "replicas",
                "clean_flag",
                "geom_flag",
            ],
        )
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    r = _pearson_safe(xs, ys)
    plot_png = out_dir / "stasep_predtotal_vs_expression.png"
    _scatter_with_fit(
        xs,
        ys,
        xlabel="Expression (Value)",
        ylabel="sTASEP predicted total ribosome count (mean over reps)",
        title=f"sTASEP predicted total vs Expression (folding={'on' if use_geometry else 'off'})",
        outpath=plot_png,
    )

    with (out_dir / "correlation_summary.txt").open("w") as fh:
        fh.write(f"rows={len(xs)}, Pearson r={r}\nplot={plot_png.name}\n")
    with (out_dir / "correlation_summary.json").open("w") as fh:
        json.dump({"n": int(xs.size), "pearson_r": r, "plot": str(plot_png), "folding": "on" if use_geometry else "off"}, fh, indent=2)

    print("\nSaved:")
    print("  Aggregated CSV:", agg_csv)
    print("  Correlation plot:", plot_png)
    print("  correlation_summary.txt / .json under:", out_dir)
    if not use_geometry:
        print("[NOTE] Folding disabled or ViennaRNA missing - geometry set to zeros.")


if __name__ == "__main__":
    main()
