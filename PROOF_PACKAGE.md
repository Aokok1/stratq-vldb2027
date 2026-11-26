# Simultaneous grouped COUNT

## Sampling and centering

Fix binary reference labels and the group partition. At each reveal, the scheduler chooses a group using the revealed history, and the next row is conditionally uniform among that group's remaining rows. Predictable quantities may use cheap features of all rows but may not depend on the unrevealed random draw order or labels. Give each group risk α_g > 0 with Σ_g α_g ≤ δ.

For one group of size N and label mean μ, at local draw s write m=N−s+1, w=N/m, and h=S_(s−1)/m, where S is the observed label sum. Let f be a predictable predictor with known remaining mean f̄ and valid lower and upper bounds f_min, f_max. Set Z=f̄+y_i−f(i), a=f̄−f_max, and R=1+f_max−f_min. Conditional uniformity gives E[Z | history, route]=wμ−h. Thus X=(Z−a)/R is in [0,1] and has conditional mean η=(wμ−h−a)/R.

## Confidence sequence

For |x|≤1, 0<λ<1, and u=λx, integration gives u−log(1+u)=u²∫₀¹ r/(1+ur) dr. The integral decreases with u, so bounding it at u=−λ yields u−log(1+u)≤x²ψ(λ), with ψ(λ)=−log(1−λ)−λ. For a predictable center p∈[0,1] and either sign σ, apply this bound to x=σ(X−p). The conditional expectation of exp(σλ(X−η)−ψ(λ)(X−p)²) is at most e^(−v)(1+v)≤1, where v=σλ(η−p). At λ=0 the factor equals one.

Multiply these factors across global steps, using factor one on draws from other groups. Conditioning on the route first and applying the tower property gives two nonnegative supermartingales starting at one. The log products at local time t are σ(B_t−A_t μ)−V_t, where A_t=Σ λw/R, B_t=Σ λ(Z+h)/R, and V_t=Σ ψ(λ)(X−p)². By Ville's inequality either sign crosses 2/α_g with probability at most α_g/2 each. Invert the two noncrossing inequalities to obtain [ (B_t−V_t−log(2/α_g))/A_t, (B_t+V_t+log(2/α_g))/A_t ] when A_t>0. Use the whole real line as the statistical interval when A_t=0.

Start from [0,1]. Intersect the previous reported interval, the current statistical interval, and the deterministic interval [S_t/N,(S_t+N−t)/N]. In real arithmetic this intersection cannot be empty on the joint coverage event. The safe binary64 path uses the stricter restart rule below: it restarts only when the conservative previous and deterministic intervals are disjoint; a conservative three-way collision is widened to their convex hull. At census always return the exact singleton. Scaling by N turns mean bounds into count bounds. The union bound over groups and signs gives coverage of at least 1−δ simultaneously over all groups and reveal times.

## Query certificates

Use the midpoint for an error guarantee of half the reported interval width on the common coverage event. Under a budget cap, report the realized widths even if they exceed the requested tolerances. For HAVING count ≥ τ, accept when L≥τ, reject when U<τ, and otherwise abstain. Strict separation of the smallest selected lower bound and largest unselected upper bound certifies top-k membership. At census a deterministic secondary sort key resolves ties. These arguments use real arithmetic and apply to a fixed query snapshot. Latency and the cost of evaluating the reference predicate are empirical questions.

## Safe range–variance gating

Let the history before a batch be $\mathcal F_{t-1}$. The selector receives only labels and cheap features revealed before that batch, the pooled predictor fitted on those labels, and feature-only group statistics. It never receives the current batch's reference labels. The default `gating_mode="safe_loo"` uses the following construction; `legacy` preserves the former in-sample selector for reproduction.

For residual $r=y-X\theta_{\rm global}$, center $r$ and the non-intercept columns $X_c$. At grid value $\kappa$, fit a Ridge coefficient on the scaled design $Z_\kappa=\kappa X_c$ with penalty $\Lambda=\lambda I$:

$$
\beta_\kappa=(Z_\kappa^\top Z_\kappa+\Lambda)^{-1}Z_\kappa^\top(r-\bar r\mathbf 1),\qquad
H_\kappa=\frac{J}{n}+Z_\kappa(Z_\kappa^\top Z_\kappa+\Lambda)^{-1}Z_\kappa^\top .
$$

The hat diagonal $h_{ii}=(H_\kappa)_{ii}$ gives the exact Ridge leave-one-out residual

$$
e_i^{\rm LOO}=\frac{r_i-\bar r-(Z_\kappa\beta_\kappa)_i}{1-h_{ii}},
\qquad v_\kappa=\frac1n\sum_i(e_i^{\rm LOO})^2 .
$$

Thus no loop over $n$ refits is needed. The pure baseline is the intercept-only LOO fit of $y$, with $R_0=1$. A nonzero candidate uses slopes $\theta_{\rm global,1:}+\beta_\kappa$ and re-centers the intercept. Its predictable score range is computed from cheap row features or the exact feature box; candidates with span greater than one are rejected rather than projected, because projection would invalidate the LOO correspondence. The range factor is $R_\kappa=1+f_{\max}-f_{\min}$.

For effective target $\epsilon$, the implemented betting-width surrogate uses

$$
\lambda_\kappa=\min\!\left(0.5,\max(10^{-4},\epsilon R_\kappa/v_\kappa)\right),
$$

$$
C_\kappa=\frac{\log(2/\alpha)R_\kappa}
{\lambda_\kappa\left(\epsilon-\psi(\lambda_\kappa)v_\kappa/(\lambda_\kappa R_\kappa)\right)}
$$

when the denominator is positive, and the legacy finite-margin fallback otherwise. The selector minimizes $C_\kappa$ over $\{0,0.1,0.25,0.5,0.75,1\}$. It returns a nonzero proxy only if $n\ge12$, the effective residual degrees of freedom $n-\operatorname{tr}H_\kappa\ge3$, and $C_\kappa\le0.95C_0$. Otherwise it returns $\kappa=0$, an intercept-only predictor and exactly zero score range. This is a predictable model-selection rule, not an additional coverage assumption.

For HAVING, the effective normalized target is clipped by the current interval's distance to the threshold and the operator's explicit indifference tolerance. For TOP-k, the boundary is the current largest unselected upper bound and the fallback is one row's normalized contribution. If the interval contains the boundary, the distance lower bound is zero and the positive floor is one quarter of the current normalized half-width; the target never exceeds the explicit fallback. Therefore `target_eps=None` no longer invokes a hard-coded 0.05 in safe mode.

## Outward-rounded binary64 intervals

`safe_rounding=True` treats stored binary64 inputs as exact reals and encloses each basic operation. With $u=2^{-53}$, a correctly rounded binary64 operation is enclosed by `nextafter(result, -inf)` and `nextafter(result, +inf)`; exact zero and one identities are preserved where applicable. The implementation applies this to products, sums, differences, divisions, dot products, remaining-feature means, the full outward interval for each normalized label, $Z$, $R$, deterministic endpoints and final affine rescaling. The group score range is enclosed before it is used in betting.

The population feature totals are not inferred by multiplying a rounded sample mean by $N$. The engine computes directed column-sum intervals for every group's cheap features using the standard reduction bound $\gamma_n\sum_i|\phi_i|\le\gamma_n n$, where $\gamma_n=nu/(1-nu)$ and all normalized features lie in $[0,1]$. These intervals are passed to the CS. For backward compatibility, a caller that supplies only `phi_mean_init` receives the analogous mean-times-total envelope; the tight engine path uses the raw-feature sum interval.

The accumulators have the directed invariants

$$
A_t^-\le A_t,\qquad B_t^-\le B_t\le B_t^+,\qquad V_t^+\ge V_t .
$$

In particular $A_t$ is rounded down after every addition and $V_t$ up. For $0\le\lambda\le0.5$, $\psi$ is enclosed using

$$
\psi(\lambda)=\sum_{k=2}^{\infty}\frac{\lambda^k}{k},\qquad
\sum_{k=98}^{\infty}\frac{\lambda^k}{k}
\le\frac{\lambda^{98}}{98(1-\lambda)} .
$$

All 96 retained terms and the tail bound are rounded outward. The log threshold is computed with an 80-digit correctly rounded Decimal logarithm. Because Decimal-to-binary64 conversion alone does not prove enclosure near a rounding boundary, the implementation first takes adjacent Decimal values on both sides of the rounded logarithm and then converts those endpoints outward. If $\ell^- \le \log(2/\alpha)\le\ell^+$, the reported statistical endpoints satisfy

$$
L_t=\frac{B_t^--(\ell^++V_t^+)}{A_t^-},\qquad
U_t=\frac{B_t^++(\ell^++V_t^+)}{A_t^-},
$$

with outward division. Hence $L_t$ is no larger and $U_t$ no smaller than the corresponding real-arithmetic endpoints. The same directed rules apply to $[S_t/N,(S_t+N-t)/N]$. At census, COUNT endpoints are restored as exact integers; non-integral mean endpoints remain outward enclosures of the exact rational values.

### Empty intersections and restarts

Let $I_{t-1}$ be the previous reported interval, $J_t$ the current statistical interval and $D_t$ the deterministic interval. The real-arithmetic intersection $I_{t-1}\cap J_t\cap D_t$ cannot be empty on the joint coverage event, because the true group mean belongs to all three. Safe mode maintains outward enclosures of every component.

A restart to $D_t$ occurs only when the conservative previous and deterministic intervals are themselves disjoint. Conservative disjointness implies exact-real disjointness. Since the true mean always belongs to $D_t$, it cannot belong to $I_{t-1}$ in that case; therefore every such restart is strictly outside the joint coverage event. If only the three-way intersection is conservatively empty, the implementation does not restart: it reports the convex hull of the conservative components. That hull contains the exact intersection whenever the latter is nonempty, and therefore preserves coverage while avoiding a floating-point-induced false restart. This is deliberately more conservative than the former unconditional reset.

The binary64 argument assumes valid outward `exact_f_bounds` supplied by the safe engine path, finite inputs, and the uniform WoR reveal law above. It does not claim correctness for arbitrary non-finite features or a caller that supplies inward-rounded score bounds.

