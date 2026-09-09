#!/usr/bin/env python3
"""
Regression tests for batch/ID correspondence.

Batching must never change which sequence a prediction belongs to. A collate
that reorders the batch (e.g. by length, for padding efficiency) while the
caller indexes results by input position silently attaches every prediction to
the wrong sequence -- and because output lengths are taken from the input list,
the results still *look* well-formed. Shape-only smoke tests cannot catch it,
so these tests check correspondence directly.

test_pad_collate_preserves_order runs without weights or a GPU.
test_batched_matches_solo needs weights under repo weights/ (skipped if absent).

Run from repo root:
  python tests/test_batch_order.py
  python -m pytest tests/test_batch_order.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from seq2ribo.data import pad_collate  # noqa: E402

WEIGHTS_DIR = REPO_ROOT / "weights"
CACHE_DIR = REPO_ROOT / "cache" / "geometry_batch_order"

# Distinct lengths (10, 8, 12 codons) so any length-based reordering shows up,
# and distinct codon composition so the profiles are distinguishable.
SEQS: List[str] = [
    "AUG" + "GCC" * 8 + "UAG",
    "AUG" + "CGA" * 6 + "UAA",
    "AUG" + "AAA" * 10 + "UGA",
]
N_STASEP_RUNS = 200


def _item(tx: str, L: int) -> dict:
    return {
        "tx": tx,
        "length": L,
        "cod_ids": torch.arange(L) % 64,
        # constant == the sequence's own length, so each row carries a fingerprint
        "sim_feat": torch.full((L,), float(L)),
        "angle_bin": torch.zeros(L, dtype=torch.long),
        "pair_bin": torch.zeros(L, dtype=torch.long),
        "bucket_idx": torch.zeros(L, dtype=torch.long),
    }


def test_pad_collate_preserves_order() -> None:
    """pad_collate must emit rows in the caller's order, not sorted by length."""
    lengths = [10, 8, 12]
    batch_data = [_item(f"seq_{i}", L) for i, L in enumerate(lengths)]
    batch = pad_collate(batch_data)

    assert batch["tx"] == [x["tx"] for x in batch_data], (
        f"pad_collate reordered the batch: {batch['tx']} != "
        f"{[x['tx'] for x in batch_data]}"
    )
    assert batch["cod_ids"].shape[1] == max(lengths)

    for i, L in enumerate(lengths):
        # row i must carry sequence i's data, unpadded to exactly its own length
        assert int(batch["mask"][i].sum()) == L, (
            f"row {i}: mask covers {int(batch['mask'][i].sum())} positions, expected {L}"
        )
        assert torch.equal(batch["sim_feat"][i, :L], torch.full((L,), float(L))), (
            f"row {i} holds another sequence's features: {batch['sim_feat'][i, :L].tolist()}"
        )
        # padding must stay out of the real region
        assert not batch["mask"][i, L:].any(), f"row {i} has mask set past its length"


def test_batched_matches_solo() -> None:
    """Predicting a list must give each sequence the same answer as alone."""
    if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.glob("*.pt")):
        print(f"SKIP: no weights in {WEIGHTS_DIR}")
        return

    from seq2ribo import Seq2Ribo

    predictor = Seq2Ribo(
        cell_line="hek293",
        weights_dir=str(WEIGHTS_DIR),
        cache_dir=str(CACHE_DIR),
    )

    for task in ("riboseq", "te"):
        random.seed(0)
        np.random.seed(0)
        batched = predictor.predict(SEQS, task=task, n_stasep_runs=N_STASEP_RUNS)

        solo = []
        for seq in SEQS:
            random.seed(0)
            np.random.seed(0)
            solo.append(predictor.predict([seq], task=task, n_stasep_runs=N_STASEP_RUNS)[0])

        assert len(batched) == len(SEQS), f"[{task}] got {len(batched)} results for {len(SEQS)} inputs"

        for i, seq in enumerate(SEQS):
            got = np.atleast_1d(np.asarray(batched[i], dtype=float))
            want = np.atleast_1d(np.asarray(solo[i], dtype=float))

            if task == "riboseq":
                # each profile must be exactly its own sequence's codon count
                assert got.shape == (len(seq) // 3,), (
                    f"[{task}] seq_{i}: got {got.shape[0]} codons, expected {len(seq) // 3}"
                )
            assert got.shape == want.shape, (
                f"[{task}] seq_{i}: batched shape {got.shape} != solo shape {want.shape}"
            )
            assert np.allclose(got, want, rtol=1e-3, atol=1e-4), (
                f"[{task}] seq_{i} got another sequence's prediction:\n"
                f"  batched: {np.round(got, 4).tolist()}\n"
                f"  solo   : {np.round(want, 4).tolist()}"
            )
        print(f"OK: {task} batched predictions match per-sequence predictions")


def main() -> int:
    errors = 0
    for fn in (test_pad_collate_preserves_order, test_batched_matches_solo):
        try:
            fn()
            print(f"OK: {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL: {fn.__name__}: {e}", file=sys.stderr)
            errors += 1
        except Exception as e:  # environment problems shouldn't read as a pass
            print(f"ERROR: {fn.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
            errors += 1
    print("\nAll batch-order tests passed." if errors == 0 else f"\n{errors} test(s) failed.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
