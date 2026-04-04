#!/usr/bin/env python3
"""
End-to-end smoke tests: all supported cell lines and tasks, multiple inputs,
Python API and CLI (run_inference.py).

Requires: working torch, mamba_ssm, weights under repo weights/, ViennaRNA optional.

Run from repo root:
  python tests/test_all_tasks_api_cli.py

Or:
  cd /path/to/seq2ribo && python -m pytest tests/test_all_tasks_api_cli.py -q
  (pytest optional; script also runs standalone via main())
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

# Repo root = parent of tests/
REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
RUN_INFERENCE = REPO_ROOT / "scripts" / "run_inference.py"
CACHE_DIR = REPO_ROOT / "cache" / "geometry_test_all_tasks"

CELL_LINES = ["hek293", "lcl", "rpe", "ipsc"]

# Short CDS-length RNAs (multiple inputs for batch-style API / FASTA CLI)
CDS_SEQS: List[str] = [
    "AUG" + "GCC" * 8 + "UAG",
    "AUG" + "CCC" * 6 + "UAA",
    "AUG" + "GGC" * 10 + "UGA",
]

UTR5_SEQS = ["AUGCUA", "GGCAUG", "AUGGGU"]
UTR3_SEQS = ["UAAUCG", "CGUAAU", "UGAUCG"]

# Fewer sTASEP runs for test speed (still exercises full pipeline)
N_STASEP_RUNS = 3


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def run_cli(
    args: List[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(RUN_INFERENCE), *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_api() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from seq2ribo import Seq2Ribo

    errors = 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for cl in CELL_LINES:
        predictor = Seq2Ribo(
            cell_line=cl,
            weights_dir=str(WEIGHTS_DIR),
            cache_dir=str(CACHE_DIR),
        )

        # --- riboseq: multiple sequences ---
        try:
            out = predictor.predict(
                CDS_SEQS,
                task="riboseq",
                n_stasep_runs=N_STASEP_RUNS,
            )
            if len(out) != len(CDS_SEQS):
                _fail(f"[API riboseq {cl}] expected {len(CDS_SEQS)} results, got {len(out)}")
                errors += 1
            else:
                for i, row in enumerate(out):
                    arr = np.asarray(row)
                    if arr.ndim != 1 or not np.isfinite(arr).all():
                        _fail(f"[API riboseq {cl}] seq {i} bad shape or non-finite")
                        errors += 1
                        break
                else:
                    _ok(f"API riboseq {cl} ({len(CDS_SEQS)} seqs)")
        except Exception as e:
            _fail(f"[API riboseq {cl}] {e}")
            errors += 1

        # --- TE CDS-only: inverse + scaled ---
        try:
            inv = predictor.predict(
                CDS_SEQS,
                task="te",
                n_stasep_runs=N_STASEP_RUNS,
            )
            scl = predictor.predict(
                CDS_SEQS,
                task="te",
                n_stasep_runs=N_STASEP_RUNS,
                return_scaled_te=True,
            )
            if len(inv) != len(CDS_SEQS) or len(scl) != len(CDS_SEQS):
                _fail(f"[API te {cl}] length mismatch")
                errors += 1
            else:
                for i, s in enumerate(scl):
                    if not (0.0 <= float(s) <= 1.0):
                        _fail(f"[API te scaled {cl}] seq {i} out of [0,1]: {s}")
                        errors += 1
                        break
                else:
                    _ok(f"API te CDS {cl} inverse+scaled ({len(CDS_SEQS)} seqs)")
        except Exception as e:
            _fail(f"[API te {cl}] {e}")
            errors += 1

        # --- TE + UTR: split lists, multiple rows ---
        try:
            inv_u = predictor.predict(
                task="te",
                use_utr=True,
                utr5_list=UTR5_SEQS,
                cds_list=CDS_SEQS,
                utr3_list=UTR3_SEQS,
                n_stasep_runs=N_STASEP_RUNS,
            )
            scl_u = predictor.predict(
                task="te",
                use_utr=True,
                utr5_list=UTR5_SEQS,
                cds_list=CDS_SEQS,
                utr3_list=UTR3_SEQS,
                n_stasep_runs=N_STASEP_RUNS,
                return_scaled_te=True,
            )
            if len(inv_u) != len(CDS_SEQS):
                _fail(f"[API te_utr {cl}] wrong length")
                errors += 1
            else:
                for i, s in enumerate(scl_u):
                    if not (0.0 <= float(s) <= 1.0):
                        _fail(f"[API te_utr scaled {cl}] seq {i} out of [0,1]")
                        errors += 1
                        break
                else:
                    _ok(f"API te+UTR {cl} ({len(CDS_SEQS)} seqs)")
        except Exception as e:
            _fail(f"[API te_utr {cl}] {e}")
            errors += 1

        # --- protein: multiple sequences ---
        try:
            pr = predictor.predict(
                CDS_SEQS,
                task="protein",
                n_stasep_runs=N_STASEP_RUNS,
            )
            if len(pr) != len(CDS_SEQS):
                _fail(f"[API protein {cl}] length mismatch")
                errors += 1
            else:
                for i, v in enumerate(pr):
                    if not isinstance(v, (float, int)) or not np.isfinite(float(v)):
                        _fail(f"[API protein {cl}] seq {i} bad scalar")
                        errors += 1
                        break
                else:
                    _ok(f"API protein {cl} ({len(CDS_SEQS)} seqs)")
        except Exception as e:
            _fail(f"[API protein {cl}] {e}")
            errors += 1

    return errors


def test_cli() -> int:
    errors = 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    w = str(WEIGHTS_DIR)
    c = str(CACHE_DIR)
    common = [
        "--weights-dir",
        w,
        "--cache-dir",
        c,
        "--n-stasep-runs",
        str(N_STASEP_RUNS),
    ]

    for cl in CELL_LINES:
        # riboseq: FASTA with multiple records
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".fa",
            delete=False,
            encoding="utf-8",
        ) as f:
            for j, seq in enumerate(CDS_SEQS):
                f.write(f">tx{j}\n{seq}\n")
            fasta_path = f.name
        out_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as outf:
                out_path = outf.name
            proc = run_cli(
                [
                    "--fasta",
                    fasta_path,
                    "--cell-line",
                    cl,
                    "--task",
                    "riboseq",
                    "--output",
                    out_path,
                    *common,
                ]
            )
            if proc.returncode != 0:
                _fail(f"[CLI riboseq {cl}] exit {proc.returncode}\n{proc.stderr}")
                errors += 1
            else:
                with open(out_path, encoding="utf-8") as rf:
                    data = json.load(rf)
                if len(data) != len(CDS_SEQS):
                    _fail(f"[CLI riboseq {cl}] json len {len(data)} != {len(CDS_SEQS)}")
                    errors += 1
                else:
                    _ok(f"CLI riboseq {cl} FASTA ({len(CDS_SEQS)} seqs)")
        except Exception as e:
            _fail(f"[CLI riboseq {cl}] {e}")
            errors += 1
        finally:
            Path(fasta_path).unlink(missing_ok=True)
            if out_path:
                Path(out_path).unlink(missing_ok=True)

        # TE CDS: single seq + output JSON
        for scaled_flag, label in [(False, "inverse"), (True, "scaled")]:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as outf:
                out_path = outf.name
            args = [
                "--seq",
                CDS_SEQS[0],
                "--cell-line",
                cl,
                "--task",
                "te",
                "--output",
                out_path,
                *common,
            ]
            if scaled_flag:
                args.append("--return-scaled-te")
            proc = run_cli(args)
            if proc.returncode != 0:
                _fail(f"[CLI te {label} {cl}] exit {proc.returncode}\n{proc.stderr}")
                errors += 1
            else:
                with open(out_path, encoding="utf-8") as rf:
                    data = json.load(rf)
                pred = data[0]["prediction"]
                if scaled_flag:
                    if not (0.0 <= float(pred) <= 1.0):
                        _fail(f"[CLI te scaled {cl}] pred {pred}")
                        errors += 1
                    else:
                        _ok(f"CLI te scaled {cl}")
                else:
                    if not isinstance(pred, (float, int)) or not np.isfinite(float(pred)):
                        _fail(f"[CLI te inverse {cl}] bad pred")
                        errors += 1
                    else:
                        _ok(f"CLI te inverse {cl}")
            Path(out_path).unlink(missing_ok=True)

        # TE + UTR
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as outf:
            out_path = outf.name
        proc = run_cli(
            [
                "--task",
                "te",
                "--use_utr",
                "--cell-line",
                cl,
                "--utr5",
                UTR5_SEQS[0],
                "--cds",
                CDS_SEQS[0],
                "--utr3",
                UTR3_SEQS[0],
                "--output",
                out_path,
                *common,
            ]
        )
        if proc.returncode != 0:
            _fail(f"[CLI te+UTR {cl}] exit {proc.returncode}\n{proc.stderr}")
            errors += 1
        else:
            with open(out_path, encoding="utf-8") as rf:
                data = json.load(rf)
            pred = data[0]["prediction"]
            if not isinstance(pred, (float, int)) or not np.isfinite(float(pred)):
                _fail(f"[CLI te+UTR {cl}] bad pred")
                errors += 1
            else:
                _ok(f"CLI te+UTR {cl}")
        Path(out_path).unlink(missing_ok=True)

        # protein
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as outf:
            out_path = outf.name
        proc = run_cli(
            [
                "--seq",
                CDS_SEQS[0],
                "--cell-line",
                cl,
                "--task",
                "protein",
                "--output",
                out_path,
                *common,
            ]
        )
        if proc.returncode != 0:
            _fail(f"[CLI protein {cl}] exit {proc.returncode}\n{proc.stderr}")
            errors += 1
        else:
            with open(out_path, encoding="utf-8") as rf:
                data = json.load(rf)
            pred = data[0]["prediction"]
            if not isinstance(pred, (float, int)) or not np.isfinite(float(pred)):
                _fail(f"[CLI protein {cl}] bad pred")
                errors += 1
            else:
                _ok(f"CLI protein {cl}")
        Path(out_path).unlink(missing_ok=True)

    return errors


def main() -> int:
    if not WEIGHTS_DIR.is_dir():
        print(f"Missing weights directory: {WEIGHTS_DIR}", file=sys.stderr)
        return 1
    if not RUN_INFERENCE.is_file():
        print(f"Missing CLI script: {RUN_INFERENCE}", file=sys.stderr)
        return 1

    print("=== Python API ===")
    api_err = test_python_api()
    print("=== CLI ===")
    cli_err = test_cli()
    total = api_err + cli_err
    if total:
        print(f"\nDone with {total} error(s).", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


def test_python_api_matrix() -> None:
    """Pytest: full API matrix (requires weights + deps)."""
    if not WEIGHTS_DIR.is_dir():
        import pytest

        pytest.skip(f"no weights at {WEIGHTS_DIR}")
    assert test_python_api() == 0


def test_cli_matrix() -> None:
    """Pytest: full CLI matrix (requires weights + deps)."""
    if not WEIGHTS_DIR.is_dir() or not RUN_INFERENCE.is_file():
        import pytest

        pytest.skip("weights or run_inference.py missing")
    assert test_cli() == 0


if __name__ == "__main__":
    sys.exit(main())
