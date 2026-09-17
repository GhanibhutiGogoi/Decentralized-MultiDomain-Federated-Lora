# No-training rank-retention diagnostic

This starts from oracle pooled rank-16 checkpoints. It tests whether communication and rank truncation preserve an already learned model, with zero optimizer steps and no training data. It cannot establish federated training success.

| Method | After initial projection (%) | After 30 cycles (%) | Extra accuracy change (pp) | Final energy retained (%) |
|---|---:|---:|---:|---:|
| ring_uniform16 | 57.227 ± 0.585 | 57.227 ± 0.585 | 0.000 ± 0.000 | 100.001 ± 0.000 |
| ring_capability4_8_16 | 48.027 ± 0.825 | 13.530 ± 2.321 | -34.497 ± 2.752 | 42.781 ± 0.646 |
| ring_reduced2_4_8 | 21.107 ± 1.082 | 5.853 ± 1.627 | -15.253 ± 0.692 | 25.779 ± 0.462 |
| central_uniform16 | 57.227 ± 0.585 | 57.227 ± 0.585 | 0.000 ± 0.000 | 100.001 ± 0.000 |
| central_capability4_8_16 | 48.027 ± 0.825 | 11.270 ± 0.918 | -36.757 ± 1.724 | 42.400 ± 0.614 |
| central_reduced2_4_8 | 21.107 ± 1.082 | 4.457 ± 0.240 | -16.650 ± 0.852 | 25.523 ± 0.378 |

Values are means ± sample SD across the paired seeds. Cycle zero already includes distributing rank projections and assembling them; subsequent cycles contain mixing and repeated projection only.

For a common SVD basis, let $d_{ik}=1[r_i\geq k]$ and $p_k=\sum_i\pi_i d_{ik}$. The initial weighted assembly's kth coefficient is $\sigma_k p_k$. Under repeated centralized mixing and receiver truncation, it becomes $\sigma_k p_k^{t+1}$ after t cycles. For a ring, $c_k(0)=\sigma_k d_k$ and $c_k(t+1)=\mathrm{diag}(d_k)P c_k(t)$; assembly reads $\pi^T c_k(t)$. A synthetic double-precision check validates both formulas against the actual projection/merge kernels.

The mechanism has a specific scope: fixed ranks, aligned singular bases, no new local learning, and model-state averaging followed by receiver truncation. On a connected graph, repeatedly deleting a component at some peers can eventually remove it across the network. The experiment does not prove that this is the only cause of the full training accuracy gap, and does not test a proposed repair.

Full raw spectra, original-basis coefficients and independent recurrence predictions, Frobenius norms, projection residuals, assembly diagnostics, test counts and input/source hashes are preserved. Tests and plotting were executed on gpu003. No privacy result is implied.
