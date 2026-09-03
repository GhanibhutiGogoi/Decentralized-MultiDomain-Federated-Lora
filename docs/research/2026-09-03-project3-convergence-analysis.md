# Convergence of ΔW-Space Gossip with Per-Round Rank Truncation

*Project 3 theory note. Every lemma below is proven here and pinned by a numerical test in `project-3-hierarchical-gossip/tests/test_convergence_lemmas.py`; the test name is given after each statement. The two theorems are proven from the lemmas by the standard consensus-distance argument. §0 says exactly what is and is not established.*

## 0. Scope, in one paragraph

The protocol implemented by `DecentralizedRunner` compresses each client's **state** every round by truncated SVD. Truncated SVD is a biased, contractive compressor, and biased compression of the state without error feedback provably leaves a non-vanishing floor. So there are two honest results, not one. **Theorem B** covers the protocol as it runs by default and shows convergence to a neighbourhood whose radius is set by the truncation tail mass ε and the step size η, scaling as ε²/η². **Theorem A** covers the protocol with error feedback switched on (`error_feedback=True`), and shows the O(1/√T) rate one expects from decentralized SGD, with an additive term of order ε² that is the irreducible approximation error of a rank-r model and does not depend on η. Both are self-contained; neither relies on quoting a theorem from the compression literature, though the technique is the standard one and the lineage is cited. What is *not* proven: any bound in which ε vanishes without error feedback, and any rate for the hard two-tier schedule finer than the windowed argument in Lemma 5(b).

## 1. Setting

$N$ clients minimise $f(x) = \frac{1}{N}\sum_{i=1}^{N} f_i(x)$ over $x \in \mathbb{R}^d$, $d = d_{\text{out}} d_{\text{in}}$, where $x = \operatorname{vec}(\Delta W)$ is the vectorised effective LoRA update of one layer. (Multiple layers are handled independently and identically.) Client $i$ can store only a rank-$r_i$ factor pair, i.e. its state lies in $\mathcal{M}_{r_i} = \{\operatorname{vec}(\Delta W) : \operatorname{rank}\Delta W \le r_i\}$.

Write $X^t \in \mathbb{R}^{N \times d}$ for the matrix whose rows are the $x_i^t$, $\bar x^t = \frac{1}{N}\sum_i x_i^t$, $J = \frac{1}{N}\mathbf{1}\mathbf{1}^\top$, and $C_r$ for truncation to rank $r$ (applied to the matrix form of each row).

**One round.** Given the mixing matrix $W^t$:

$$
\begin{aligned}
\tilde x_i^t &= x_i^t - \eta\, g_i^t && \text{local step, } \mathbb{E}\,g_i^t = \nabla f_i(x_i^t) \\
y_i^t &= \textstyle\sum_j W^t_{ij}\, \tilde x_j^t \;[+\, m_i^t] && \text{mixing (memory added only with feedback)} \\
x_i^{t+1} &= C_{r_i}(y_i^t) && \text{truncation to what the client can store} \\
m_i^{t+1} &= y_i^t - x_i^{t+1} && \text{residual, kept only with feedback}
\end{aligned}
$$

Define the compression error $e_i^t = x_i^{t+1} - y_i^t$ (so $m_i^{t+1} = -e_i^t$), $E^t$ its matrix, and the consensus distance $\Xi^t = \|(I - J)X^t\|_F^2 = \sum_i \|x_i^t - \bar x^t\|^2$.

## 2. Assumptions

- **A1 (smoothness).** Each $f_i$ is $L$-smooth.
- **A2 (bounded second moment).** $\mathbb{E}\|g_i^t\|^2 \le G^2$, and $\mathbb{E}\|g_i^t - \nabla f_i(x_i^t)\|^2 \le \sigma^2$.
- **A3 (mixing).** Each $W^t$ is symmetric and doubly stochastic. There are a window $L \ge 1$ and $\rho \in (0, 1]$ such that every product $P_t = W^{t+L-1}\cdots W^{t}$ satisfies $\|P_t - J\|_2 \le 1 - \rho$. For a time-invariant $W$ with second-largest eigenvalue magnitude $|\lambda_2|$, this holds with $L = 1$ and $\rho = 1 - |\lambda_2|$, the spectral gap returned by `mixing.spectral_gap`. Metropolis–Hastings matrices satisfy the doubly stochastic condition exactly; Sinkhorn-projected affinity matrices satisfy it to the projection tolerance (`SINKHORN_TOL = 1e-10`), so every "exact" conservation statement below holds to that tolerance for them — a per-round mean drift of at most $10^{-10}\max_i|x_i|$, which is far below any other term in the bounds.
- **A4 (compression).** $\|C_r(x) - x\|^2 \le (1 - \delta)\|x\|^2$ for some $\delta \in (0, 1]$. Lemma 2 shows truncated SVD satisfies this with $\delta = r / \min(d_{\text{out}}, d_{\text{in}})$.
- **A5 (tail mass).** $\varepsilon^2 := \sup_t \frac{1}{N}\mathbb{E}\|E^t\|_F^2$ is finite. This is the quantity the runner logs as `mean_tail_mass` (relative form). It is small precisely when the iterates stay close to rank-$r$ matrices — the LoRA hypothesis. Nothing below assumes it is small; the theorems are stated in terms of it.

## 3. Lemmas

### Lemma 1 — mixing contracts consensus error at rate $|\lambda_2|$

*Let $W$ be symmetric doubly stochastic with eigenvalues $1 = \lambda_1 \ge |\lambda_2| \ge \cdots$. Then $JW = WJ = J$, and for every $X \in \mathbb{R}^{N\times d}$,*
$$\|(W - J)X\|_F \le |\lambda_2|\,\|(I - J)X\|_F.$$

*Proof.* $W$ is symmetric, so $W = \sum_k \lambda_k v_k v_k^\top$ with orthonormal $v_k$. Doubly stochastic gives $W\mathbf{1} = \mathbf{1}$ and $\mathbf{1}^\top W = \mathbf{1}^\top$, so $v_1 = \mathbf{1}/\sqrt N$ with $\lambda_1 = 1$, and $J = v_1 v_1^\top$. Hence $JW = WJ = J$ and $W - J = \sum_{k \ge 2}\lambda_k v_k v_k^\top$, an operator of norm $|\lambda_2|$ that annihilates $v_1$. Since $(W - J)J = J - J = 0$, we have $(W - J)X = (W - J)(I - J)X$, and the bound follows from $\|(W-J)\|_2 = |\lambda_2|$. $\blacksquare$

Tests: `test_lemma1_mixing_contraction`, `test_lemma1_bound_is_tight_on_the_second_eigenvector`, `test_lemma1_JW_equals_WJ_equals_J`.

### Lemma 2 — truncated SVD is a $\delta$-contractive compressor, and the metric projection onto $\mathcal{M}_r$

*Let $X \in \mathbb{R}^{m\times n}$ have singular values $\sigma_1 \ge \cdots \ge \sigma_R \ge 0$, $R = \min(m,n)$, and $C_r(X) = \sum_{k \le r}\sigma_k u_k v_k^\top$. Then*
$$\|C_r(X) - X\|_F^2 = \sum_{k > r}\sigma_k^2 \;\le\; \Big(1 - \frac{r}{R}\Big)\|X\|_F^2,$$
*with equality on the right iff all singular values are equal, and $C_r(X)$ minimises $\|Y - X\|_F$ over all $Y$ of rank $\le r$.*

*Proof.* $X - C_r(X) = \sum_{k>r}\sigma_k u_k v_k^\top$, and the $u_k v_k^\top$ are orthonormal in the Frobenius inner product, giving the equality. The sum of the $R - r$ smallest of $R$ non-negative numbers is at most $\frac{R-r}{R}$ of their total, giving the inequality, with equality iff all are equal. Optimality over rank-$\le r$ matrices is the Eckart–Young–Mirsky theorem. $\blacksquare$

So A4 holds with $\delta = r/R$ uniformly, and the *realised* contraction on a particular $X$ is $\delta(X) = \sum_{k\le r}\sigma_k^2/\|X\|_F^2 = 1 - (\text{tail mass})$. The worst case is a flat spectrum; a rapidly decaying spectrum — the LoRA hypothesis — gives $\delta(X)$ close to 1. Note that `merge.factorize_delta` returns factors scaled by $\sqrt{\alpha/r}$ so that the forward pass $\frac{\alpha}{r}BA$ reproduces $C_r(X)$ exactly; the compressor is $C_r$, not the factors.

Tests: `test_lemma2_compressor_bound_and_exact_tail_identity`, `test_lemma2_bound_is_tight_for_a_flat_spectrum`, `test_lemma2_is_the_metric_projection_onto_rank_r`.

### Lemma 3 — products, sandwiches, and the two-tier schedule

*(a) A product of doubly stochastic matrices is doubly stochastic. (b) If $A, B$ are symmetric then $ABA$ is symmetric. (c) The two-tier matrix $W = W_{\text{intra}} W_{\text{bridge}} W_{\text{intra}}$ of `hierarchical.two_tier_mixing` is symmetric doubly stochastic. (d) Under complete-graph intra-cluster mixing, $W_{\text{intra}}$ is idempotent, so the product over one period of length $L =$ `bridge_every` equals the bridge-round matrix, and A3 holds with that $L$ and $\rho = 1 - |\lambda_2(W)|$. (e) Between bridges the matrix is block-diagonal with one block per cluster, so $1$ is an eigenvalue of multiplicity $K$ and the per-round spectral gap is $0$; the gap lives in the window product.*

*Proof.* (a) $\mathbf{1}^\top AB = \mathbf{1}^\top B = \mathbf{1}^\top$, $AB\mathbf{1} = A\mathbf{1} = \mathbf{1}$, and entries stay non-negative. (b) $(ABA)^\top = A^\top B^\top A^\top = ABA$. (c) $W_{\text{intra}}$ is block-diagonal with symmetric doubly stochastic Metropolis–Hastings blocks, hence symmetric doubly stochastic; $W_{\text{bridge}}$ embeds a Sinkhorn output -- symmetric exactly, doubly stochastic to the projection tolerance (A3) -- into an identity, hence the same; apply (a) and (b). (d) Metropolis–Hastings on a complete graph $K_m$ gives every weight $1/m$, i.e. the block $J_m$, and $J_m^2 = J_m$; so $W_{\text{intra}}^{L-1} = W_{\text{intra}}$ and $W\, W_{\text{intra}}^{L-1} = W_{\text{intra}}W_{\text{bridge}}W_{\text{intra}}W_{\text{intra}} = W$. For $\rho > 0$: $W$ has a strictly positive diagonal (each factor does, and products of non-negative matrices with positive diagonals have positive diagonals) and its support graph is connected (members of a cluster are connected through $J_m$, clusters through the complete representative graph with strictly positive Sinkhorn weights), so it is primitive and Perron–Frobenius gives $|\lambda_2| < 1$. (e) Each diagonal block has eigenvalue $1$ with its own eigenvector. $\blacksquare$

Tests: `test_lemma3_product_of_doubly_stochastic_is_doubly_stochastic`, `test_lemma3_sandwich_is_symmetric`, `test_lemma3_two_tier_window_has_positive_gap_iff_bridge_in_window`, `test_lemma3_eigenvalue_one_multiplicity_equals_number_of_clusters_between_bridges`, and `test_complete_intra_mixing_makes_the_window_product_equal_the_bridge_matrix` in `test_hierarchical.py`.

### Lemma 4 — mean dynamics

*Without feedback:*
$$\bar x^{t+1} = \bar x^t - \eta\,\bar g^t + \bar e^t, \qquad \bar e^t = \tfrac{1}{N}\textstyle\sum_i e_i^t .$$
*With feedback, define the virtual iterate $v_i^t = x_i^t + m_i^t$. Then*
$$\bar v^{t+1} = \bar v^t - \eta\,\bar g^t \quad\text{exactly.}$$

*Proof.* From $JW = J$ (Lemma 1), $\bar y^t = \bar x^t - \eta\bar g^t$ without feedback, and $x^{t+1}_i = y_i^t + e_i^t$ gives the first identity. With feedback, $y_i^t = \sum_j W_{ij}\tilde x_j^t + m_i^t$, so $\bar y^t = \bar x^t - \eta\bar g^t + \bar m^t = \bar v^t - \eta\bar g^t$; and $v_i^{t+1} = x_i^{t+1} + m_i^{t+1} = x_i^{t+1} + (y_i^t - x_i^{t+1}) = y_i^t$, so $\bar v^{t+1} = \bar y^t$. $\blacksquare$

The point: with feedback the compression error is *bookkept*, not discarded, and the mean of the virtual iterates is exact averaged SGD. Without it the mean is kicked by $\bar e^t$ every round, and $\bar e^t$ does not shrink with $\eta$.

Tests: `test_lemma4_without_feedback_the_mean_shifts_by_the_mean_compression_error`, `test_lemma4_with_feedback_the_virtual_mean_is_exactly_conserved_over_many_rounds`, and in `test_runner.py` `test_error_feedback_conserves_the_virtual_mean`, `test_error_feedback_memory_is_exactly_the_residual`.

### Lemma 5 — the consensus recursion

*(a) Under A2–A5 with $L = 1$,*
$$\sqrt{\Xi^{t+1}} \le (1-\rho)\sqrt{\Xi^t} + (1-\rho)\,\eta\,\|G^t\|_F + \|(I-J)E^t\|_F$$
*(with $E^t$ replaced by $M^t - M^{t+1}$ under feedback), and consequently*
$$\mathbb{E}\,\Xi^{t} \le (1-\rho)^t\,\Xi^0 + \frac{2N}{\rho^2}\big(\eta^2 G^2 + c\,\varepsilon^2\big), \qquad c = 1 \text{ (plain)},\; c = 4 \text{ (feedback)}.$$
*(b) Under A3 with a window $L > 1$ the same bound holds with $\rho$ the window gap and the bracket multiplied by $L^2$.*

*Proof.* (a) $(I-J)X^{t+1} = (I-J)Y^t + (I-J)E^t$, and $(I-J)Y^t = (W-J)\tilde X^t$ (plain) by $JW = J$; with feedback $(I-J)Y^t = (W-J)\tilde X^t + (I-J)M^t$ and $(I-J)E^t = -(I-J)M^{t+1}$, which combine to the stated replacement. Lemma 1 gives $\|(W-J)\tilde X^t\|_F \le (1-\rho)\|(I-J)\tilde X^t\|_F \le (1-\rho)(\sqrt{\Xi^t} + \eta\|G^t\|_F)$, using $\|(I-J)\|_2 \le 1$. Square with Young's inequality $(a+b)^2 \le (1+\beta)a^2 + (1+\beta^{-1})b^2$, $\beta = \rho/(1-\rho)$, so that $(1+\beta)(1-\rho)^2 = 1-\rho$ and $1+\beta^{-1} = 1/\rho$; then $(b_1 + b_2)^2 \le 2b_1^2 + 2b_2^2$ and take expectations with $\mathbb{E}\|G^t\|_F^2 \le NG^2$ and $\mathbb{E}\|E^t\|_F^2 \le N\varepsilon^2$ (or $\|M^t - M^{t+1}\|_F^2 \le 4N\varepsilon^2$). Unroll the geometric recursion. (b) Apply the same argument to the window product $P_t$, absorbing the $L$ within-window gradient and compression terms by the triangle inequality, which multiplies each by at most $L$ and the bracket by $L^2$. $\blacksquare$

Tests: `test_lemma5_one_step_consensus_recursion_holds` ($\eta = 0$), `test_lemma5_one_step_recursion_holds_with_gradient_steps` ($\eta > 0$, on the runner with real local steps), `test_lemma5_unrolled_bound_holds_with_compression_only` (the unrolled steady-state form), `test_lemma5_without_compression_the_recursion_is_geometric`.

### Lemma 6 — the self-weight floor

*For $W_0$ symmetric doubly stochastic and $w_{\min} \in [0,1)$, $W = w_{\min} I + (1 - w_{\min})W_0$ is symmetric doubly stochastic with $W_{ii} \ge w_{\min}$ and $\lambda_k(W) = w_{\min} + (1-w_{\min})\lambda_k(W_0)$, hence spectral gap $\rho(W) = (1-w_{\min})\rho(W_0)$.*

*Proof.* Convex combinations of symmetric doubly stochastic matrices are symmetric doubly stochastic; the eigenvalue map is affine because $I$ and $W_0$ share eigenvectors. $\blacksquare$

This is the exact price of never fully trusting peers: the floor that protects against a dissimilar or hostile neighbour slows consensus by the factor $(1 - w_{\min})$. Test: `test_lemma6_floor_shifts_eigenvalues_affinely`; also `test_self_weight_floor_scales_the_spectral_gap_exactly` in `test_hierarchical.py`.

## 4. Theorems

Both proofs use one standard inequality, stated once. For any point $z$ and the client iterates $x_i$,
$$-\Big\langle \nabla f(z),\, \tfrac{1}{N}\textstyle\sum_i \nabla f_i(x_i)\Big\rangle \;\le\; -\tfrac12\|\nabla f(z)\|^2 + \tfrac{L^2}{2}\cdot\tfrac{1}{N}\textstyle\sum_i\|x_i - z\|^2, \tag{$\star$}$$
which follows from writing $\frac1N\sum_i\nabla f_i(x_i) = \nabla f(z) + \Delta$ with $\|\Delta\|^2 \le \frac{L^2}{N}\sum_i\|x_i - z\|^2$ by A1 and Jensen, and $\langle \nabla f(z), \Delta\rangle \ge -\frac12\|\nabla f(z)\|^2 - \frac12\|\Delta\|^2$.

### Theorem A — with error feedback

*Under A1–A5 with `error_feedback=True`, for any constant $\eta \le 1/L$,*
$$\frac1T\sum_{t<T}\mathbb{E}\|\nabla f(\bar v^t)\|^2 \;\le\; \frac{2\,(f(\bar v^0) - f^\star)}{\eta T} \;+\; L\eta\Big(\frac{\sigma^2}{N} + G^2\Big) \;+\; \frac{4L^2}{\rho^2}\,\eta^2 G^2 \;+\; 2L^2\Big(1 + \frac{8}{\rho^2}\Big)\varepsilon^2 \;+\; \frac{2L^2\,\Xi^0}{N\rho T} .$$
*With $\eta = 1/\sqrt{T}$ the first three terms are $O(1/\sqrt T)$, leaving an additive $O(L^2\varepsilon^2/\rho^2)$ that does not depend on $\eta$ or $T$.*

*Proof.* By Lemma 4 the virtual mean follows $\bar v^{t+1} = \bar v^t - \eta\bar g^t$ exactly, so by A1,
$f(\bar v^{t+1}) \le f(\bar v^t) - \eta\langle\nabla f(\bar v^t), \bar g^t\rangle + \frac{L\eta^2}{2}\|\bar g^t\|^2.$
Take expectations and apply $(\star)$ with $z = \bar v^t$. The deviation term is
$\frac1N\sum_i\|x_i^t - \bar v^t\|^2 \le \frac2N\sum_i\|x_i^t - \bar x^t\|^2 + 2\|\bar x^t - \bar v^t\|^2 = \frac2N\Xi^t + 2\|\bar m^t\|^2 \le \frac2N\Xi^t + 2\varepsilon^2,$
using $\|\bar m^t\|^2 \le \frac1N\sum_i\|m_i^t\|^2 \le \varepsilon^2$ (A5, since $m_i^{t} = -e_i^{t-1}$). Also $\mathbb{E}\|\bar g^t\|^2 \le \sigma^2/N + G^2$ by A2. Hence
$\mathbb{E}f(\bar v^{t+1}) \le \mathbb{E}f(\bar v^t) - \frac\eta2\mathbb{E}\|\nabla f(\bar v^t)\|^2 + \eta L^2\big(\tfrac1N\mathbb{E}\Xi^t + \varepsilon^2\big) + \frac{L\eta^2}{2}\big(\tfrac{\sigma^2}{N} + G^2\big).$
Sum over $t < T$ and divide by $\eta T/2$: the first two displayed terms appear directly, and the consensus term becomes $2L^2\cdot\frac{1}{T}\sum_t \frac1N\mathbb{E}\Xi^t + 2L^2\varepsilon^2$. Insert Lemma 5(a) with $c = 4$: $\frac1N\mathbb{E}\Xi^t \le \frac{(1-\rho)^t}{N}\Xi^0 + \frac{2}{\rho^2}(\eta^2G^2 + 4\varepsilon^2)$, whose transient sums to at most $\Xi^0/(N\rho T)$. Collecting: $2L^2\cdot\frac{2}{\rho^2}\eta^2G^2 = \frac{4L^2}{\rho^2}\eta^2G^2$, and $2L^2\cdot\frac{8}{\rho^2}\varepsilon^2 + 2L^2\varepsilon^2 = 2L^2(1 + \frac{8}{\rho^2})\varepsilon^2$, plus the transient $\frac{2L^2\Xi^0}{N\rho T}$. $\blacksquare$

**Reading it.** The floor is $O(L^2\varepsilon^2/\rho^2)$, proportional to the tail mass of the iterates. Under the LoRA hypothesis — the optimum is (close to) rank $r$ — the tail mass near the optimum is (close to) zero and the floor vanishes; when the optimum is genuinely higher rank, the floor is the irreducible error of a rank-$r$ model, which no algorithm storing rank-$r$ states can beat. Feedback also gives a second guarantee for free: by Lemma 4 the virtual mean itself converges as unconstrained averaged SGD, so the *stored* states track the best rank-$r$ approximation of an iterate that is heading to the true optimum. Test: `test_theorem_rank_deficient_has_a_floor_that_feedback_lowers` (feedback lands within $1.5\times$ the irreducible error). The test is qualitative -- it pins the existence and ordering of the floors, not the constants; the constants were checked by independent re-derivation.

### Theorem B — without error feedback (the default protocol)

*Under A1–A5 with `error_feedback=False`, for any constant $\eta \le 1/L$,*
$$\frac1T\sum_{t<T}\mathbb{E}\|\nabla f(\bar x^t)\|^2 \;\le\; \frac{4\,(f(\bar x^0) - f^\star)}{\eta T} \;+\; 4L\eta\Big(\frac{\sigma^2}{N} + G^2\Big) \;+\; \frac{4L^2}{\rho^2}\big(\eta^2G^2 + \varepsilon^2\big) \;+\; \frac{4\varepsilon^2}{\eta^2} \;+\; \frac{4L\varepsilon^2}{\eta} \;+\; \frac{2L^2\,\Xi^0}{N\rho T}.$$

*Proof.* By Lemma 4, $\bar x^{t+1} = \bar x^t - \eta\bar g^t + \bar e^t$ with $\|\bar e^t\|^2 \le \varepsilon^2$. By A1,
$f(\bar x^{t+1}) \le f(\bar x^t) + \langle\nabla f(\bar x^t), -\eta\bar g^t + \bar e^t\rangle + \frac L2\|{-\eta\bar g^t} + \bar e^t\|^2.$
The gradient term is handled by $(\star)$ with $z = \bar x^t$, deviation $\frac1N\Xi^t$. The bias term is bounded by Young with a deliberately small constant, $\langle\nabla f, \bar e\rangle \le \frac\eta4\|\nabla f\|^2 + \frac1\eta\|\bar e\|^2$ — the constant must be smaller than the $\frac\eta2$ won by $(\star)$ or the descent is cancelled entirely. The quadratic term is $\le L\eta^2\|\bar g\|^2 + L\|\bar e\|^2$. Together,
$\mathbb{E}f(\bar x^{t+1}) \le \mathbb{E}f(\bar x^t) - \frac\eta4\mathbb{E}\|\nabla f(\bar x^t)\|^2 + \frac{\eta L^2}{2N}\mathbb{E}\Xi^t + \frac{\varepsilon^2}{\eta} + L\eta^2\big(\tfrac{\sigma^2}{N}+G^2\big) + L\varepsilon^2.$
Sum, divide by $\eta T/4$, insert Lemma 5(a) with $c = 1$; the transient $(1-\rho)^t\Xi^0$ sums to at most $\Xi^0/(N\rho T)$ and gives the last term. $\blacksquare$

**Reading it.** The dominant floor is $4\varepsilon^2/\eta^2$. It *grows* as the step size shrinks, so the usual $\eta \propto 1/\sqrt T$ schedule does not give convergence: each round the truncation kicks the mean by up to $\varepsilon$ regardless of $\eta$, and a smaller step cannot outrun a fixed kick. The protocol converges to a neighbourhood, and the neighbourhood is governed by $\varepsilon/\eta$ — the tail mass measured in units of the step. It is a good protocol exactly when the LoRA hypothesis holds strongly enough that $\varepsilon \ll \eta G$. Test: `test_theorem_rank_deficient_has_a_floor_that_feedback_lowers` (the plain protocol settles strictly above the irreducible error; feedback settles below it).

### Corollary — what error feedback buys, quantitatively

The two floors are $O(L^2\varepsilon^2/\rho^2)$ with feedback and $O(\varepsilon^2/\eta^2)$ without. Their ratio is $O(\rho^2/(L^2\eta^2))$: for small steps feedback is better by a factor that grows like $1/\eta^2$, and for any step it is better whenever $\eta < \rho/L$. This is a testable prediction, not a decoration — `test_theorem_rank_deficient_has_a_floor_that_feedback_lowers` is its smallest instance, and the natural experiment is the optional feedback arm of Experiment 07 in the Phase 4 plan. The cost is one dense $d_{\text{out}}\times d_{\text{in}}$ residual per LoRA layer per client, trivial on the fc-only testbed and prohibitive for LoRA in every block; that trade-off is why feedback is an option and not the default.

## 5. What this does and does not establish

**Established.** Lemmas 1--4, 5(a) and 6 with proofs and numerical pins; Lemma 5(b) as a sketch (the $L^2$ window factor is stated, not worked, and has no numerical pin); two self-contained theorems by the consensus-distance argument; the reduction of the periodic two-tier schedule to a single-matrix analysis under complete-graph intra mixing (Lemma 3d); the exact eigenvalue cost of the self-weight floor (Lemma 6).

**Assumed, not shown.** That the LoRA hypothesis holds on real tasks — i.e. that $\varepsilon$ is small — is an empirical claim about the data, which is exactly what the runner's `mean_tail_mass` log exists to measure. Bounded second moment (A2) is the standard but strong assumption of this literature; replacing it with a bounded-heterogeneity assumption changes constants, not structure.

**Not established.** Any vanishing bound without feedback; any rate for the two-tier schedule tighter than the $L^2$ factor of Lemma 5(b) when the intra-cluster topology is not complete; anything about affinities that are *discovered* rather than given — the mixing matrices here are inputs, and the theorems hold for whatever symmetric doubly stochastic matrices the discovery produces, but say nothing about whether discovery produces good ones. That is gate G1 and is settled empirically.

## 6. Lineage

The consensus-distance technique is that of Lian et al. (NeurIPS 2017) for decentralized parallel SGD; the compressor framework of biased $\delta$-contractive operators is from Stich, Cordonnier & Jaggi (NeurIPS 2018) and Beznosikov, Horváth, Richtárik & Safaryan (JMLR 2023); the combination of gossip with compressed communication and memory is CHOCO-SGD, Koloskova, Stich & Jaggi (ICML 2019). What is specific here is that the compressor acts on the *state* rather than on a communicated difference — forced by clients that can only hold rank-$r$ factors — which is why Theorem B has a floor that the communication-compression setting does not, and why Lemma 4's exact virtual-mean identity is what rescues Theorem A. Metropolis–Hastings weights for fastest mixing follow Xiao & Boyd (2004); Eckart–Young–Mirsky is from 1936/1960.
