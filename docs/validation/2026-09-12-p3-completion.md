# Project 3 completion validation — 12 September 2026

The starting revision is `8c22e76`. All execution used the isolated snapshot on the documented SSH GPU host, with Python 3.10.12 and torch 2.3.0+cu121. A V100 was used for real-data training and feature extraction.

The complete suite after adding the benchmark, signatures, report generator and cache checks passed **375 tests in 10.26 seconds**. This covers the existing 352 kernel/theory tests and new checks for paired initialization/splits, shuffled topology, heterogeneous factor baseline, exact operational payloads, output safety, label-independent clustering, gauge-invariant signatures, cache identity, duplicate-evidence rejection and honest paired aggregation.

Run command from Project 3:

```bash
PYTHONPATH=~/pystubs OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ~/ahlora-venv/bin/python -m pytest tests/ -q --no-header -p no:warnings -rf
```

The two-round, seed-42 smoke battery completed all four methods on all 50,000 training and 10,000 test samples. Before that, an older cache format was correctly rejected before training; its failed manifest was preserved, and the cache was regenerated with the final source/weights/transform/tensor identity. Neither smoke results nor numerical unit tests establish algorithmic superiority.

The measured three-seed runs and their raw data are indexed separately in `docs/artifacts/claude_handoff.json`. All methods use identical frozen eval-mode features, partitions and initial states within each seed. Communication counts are simulated payloads; these experiments do not measure a distributed network transport implementation.
