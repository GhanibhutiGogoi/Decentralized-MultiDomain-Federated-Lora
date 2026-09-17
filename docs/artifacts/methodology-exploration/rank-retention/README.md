# Repeated rank projection can erase an already learned adapter

The completed `attempt01/` diagnostic initializes every peer from the saved, conventional pooled rank-16 checkpoints. It performs **zero optimizer steps and zero training-example exposures**. Its purpose is to isolate information loss caused by model-state mixing followed by receiver rank truncation. These oracle-initialized results are not a federated training success result.

All 18 conditions completed on gpu003: seeds 42/43/44, six topology/rank conditions, 30 mixing cycles and an initial projection checkpoint. Every checkpoint evaluates the full 10,000-example CIFAR-100 test set using the same frozen feature cache as the corrected integrated experiment. The source script is `project-3-hierarchical-gossip/experiments/diagnose_rank_retention.py`; the isolated runtime started from repository snapshot `eed323a`, plus this new diagnostic. No original experiment sources or artifacts were overwritten.

| Operation | Initial projected accuracy | Accuracy after 30 cycles |
|---|---:|---:|
| Ring, uniform rank 16 | 57.227 ± 0.585% | 57.227 ± 0.585% |
| Ring, ranks 4/8/16 | 48.027 ± 0.825% | 13.530 ± 2.321% |
| Ring, ranks 2/4/8 | 21.107 ± 1.082% | 5.853 ± 1.627% |
| Central, uniform rank 16 | 57.227 ± 0.585% | 57.227 ± 0.585% |
| Central, ranks 4/8/16 | 48.027 ± 0.825% | 11.270 ± 0.918% |
| Central, ranks 2/4/8 | 21.107 ± 1.082% | 4.457 ± 0.240% |

Values are mean ± sample SD across seeds. Rank patterns repeat across 15 clients. Ring conditions use the actual corrected experiment's topology and sample-weighted reversible Metropolis matrix. Central controls replace this with full sample-weighted averaging. Initial projected accuracy already includes distributing truncated adapters and assembling their weighted mean. The subsequent losses are caused by repeated communication/projection without new training. The tiny approximately 0.001% energy increase reported for uniform-rank controls is floating-point accumulation, not learned information or an accuracy improvement.

The mechanism is explicit. For a common SVD basis, write the original adapter as $X=\sum_k\sigma_k u_kv_k^T$ and let $d_{ik}=1[r_i\geq k]$. With normalized sample weights $\pi_i$, define $p_k=\sum_i\pi_i d_{ik}$. The first weighted assembly has coefficient $\sigma_k p_k$. Repeating centralized mixing and rank truncation gives $\sigma_k p_k^{t+1}$ after $t$ cycles. Any component retained by only some clients therefore shrinks repeatedly.

On the ring, the corresponding recurrence is $c_k(0)=\sigma_k d_k$, $c_k(t+1)=\operatorname{diag}(d_k)P c_k(t)$, with assembled coefficient $\pi^T c_k(t)$. Because the support sets are nested by rank and coefficients remain ordered, the same singular basis is retained in this constructed diagnostic. On a connected graph, the deleted nodes act as loss points for a component; if at least one client cannot retain it, the restricted transition matrix loses that component asymptotically. The components every client can retain survive. This explains convergence toward the minimum supported rank in this setting.

`synthetic_formula_check.json` validates the central closed form and ring recurrence against the actual double-precision projection/mixing kernels. Every real-data cycle also checks observed common-basis coefficients against the independent scalar recurrence. Uniform-rank accuracy preservation controls, saved-checkpoint endpoint reproduction, full-test cache hashes, source hashes, actual versus predicted coefficients, Frobenius norms, spectra, projection residuals, and assembly diagnostics are retained in the raw records.

The result identifies a strong failure mechanism in repeatedly truncating and averaging **complete adapter states**. It does not prove that projection is the only source of the end-to-end training gap: uniform-rank decentralized training also failed in the original experiment. A repair must be tested separately, and its communication and memory costs must remain explicit. The rank-16 evaluation assembly here measures retained information; in the all-ranks-at-most-8 condition it is not a claim that a participating peer can deploy rank 16 within its training capability.

Reproduction on gpu003, with a fresh output directory:

```sh
cd ~/ahlora-exploration-20260915
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH=~/pystubs:$PWD/project-3-hierarchical-gossip \
~/ahlora-venv/bin/python \
  project-3-hierarchical-gossip/experiments/diagnose_rank_retention.py \
  --checkpoints ~/ahlora-runs/integrated-corrected-full-20260915 \
  --feature-cache ~/ahlora-data/features-v2/cifar100-resnet18-imagenet1k-v1-resize224-v1 \
  --output ~/ahlora-runs/methodology-exploration-20260915/rank-retention/NEW_ATTEMPT \
  --cycles 30 --device cuda:0
```

Read `attempt01/REPORT.md` and `summary.json` for generated results, `per_seed.csv` for unaggregated endpoints, and `seed*_*.json` for the complete 31-checkpoint trajectories. `rank_retention_curves.{pdf,png}` shows accuracy, energy and norm preservation. `rank_retention_coefficients.{pdf,png}` shows which original singular components disappear. All plots were generated remotely and inspected after retrieval. No paper edits or commits were made for this exploration.
