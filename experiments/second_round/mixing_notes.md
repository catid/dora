# What the DoRA/NoRA mixtures change

Use conventional factors $A\in\mathbb R^{r\times d_{in}}$,
$B\in\mathbb R^{d_{out}\times r}$, and frozen weight $W_0$.
For input column $j$, the implemented normalization is
$\hat a_j=a_j/\max(\lVert a_j\rVert_2,\epsilon)$, with
$\epsilon=10^{-12}$. The statements about unit columns below apply outside
this epsilon clamp. Inside the clamp, columns can have subunit norm, so these
are not global impossibility claims about every floating-point parameter value.

**The normalized low-rank branch has a constraint.** At rank one,
$\Delta_j=B\hat a_j=\pm B$: every nonzero column has the same norm. For a
target $\Delta_j=u c_j$, with unit $u$ and positive amplitudes $c_j$,
the best common amplitude is $\bar c$, giving the relative squared error

$$
\frac{\sum_j(c_j-\bar c)^2}{\sum_j c_j^2}.
$$

At full branch rank $r$, write a target as $\Delta=UV$. Any full-rank
factorization with the same column space has $B=UR$, with invertible $R$.
Unit down-factor columns require
$v_j^T H v_j=1$ for every $j$, where $H=R^{-T}R^{-1}$ is positive
definite. Arbitrary column amplitudes need not lie on such a shared ellipsoid.

An unused rank direction changes this conclusion. Every rank-$q$ update
$UV$, with $q<r$, admits a unit-column representation: choose
$c>\max_j\lVert v_j\rVert$, use $B=[cU,0]$, and set
$\hat a_j=[v_j/c,\sqrt{1-\lVert v_j/c\rVert^2}]^T$, padding with zeros
if needed. The extra component lies in the nullspace of $B$. This is why the
teacher grid separates direction ranks $q=r-1$ and $q=r$.

**That branch argument does not establish a whole-model limitation for the
combination.** DoRA gives

$$
W_{eff}=D(W_0+B\hat A),\qquad
D_{ii}=\frac{m_i}{\lVert(W_0+B\hat A)_{i,:}\rVert_2},\qquad
W_{eff}-W_0=(D-I)W_0+DB\hat A.
$$

The first term can have high rank. Learned magnitudes also change the frozen
base contribution, so a bound for $B\hat A$ alone cannot prove that the full
combination cannot fit a target. The implementation detaches the denominator
during backpropagation, while retaining NoRA's normalization derivative. A
nonzero error after 800 steps is an observed finite-budget result, not a proof
of the best possible combined-model error.

**Learned gains restore an existing DoRA factor family.** The gain variant uses
$C=\hat A\,\mathrm{diag}(\exp\ell)$. Every nonzero ordinary down-factor
column $c_j$ can be written as
$c_j=(c_j/\lVert c_j\rVert)\exp(\log\lVert c_j\rVert)$; zero columns
are also allowed by the implemented zero-vector case. Conversely, every such
$C$ is an ordinary unconstrained DoRA down factor. At fixed rank this is an
amplitude/direction reparameterization of DoRA, not a fundamentally larger
weight family than DoRA. It changes gradients, initialization scales and
optimizer behavior, and stores $d_{in}$ additional parameters compared with
the original combination. Equal rank is not equal stored parameter count.

**Slower magnitude learning is an optimizer ablation.** The MLR variant keeps
the exact same forward and parameter family, using
$\eta_m=\gamma\eta_{A,B}$, with $\gamma\in\{0.1,0.01\}$.
All methods omit magnitude/gain weight decay. Four validation configurations
are allowed per method; the MLR search trades two base rates for two magnitude
multipliers. These limited searches and one training horizon do not establish
a generally optimal magnitude schedule.

The [matched-rate validation diagnostic](matched_magnitude.py) holds factor LR,
rank, tuning seed and trial budget fixed using existing trials. It also
generally disfavors slower magnitudes on the teacher and retrieval grids:
$\gamma=0.1$ improves 14/56 teacher pairs and $\gamma=0.01$ improves 6/56;
each improves 1/4 NFCorpus validation pairs, 2/4 Aircraft validation pairs,
and 0/2 COGS validation pairs. The complete diagnostic contains 132 comparisons
and fingerprints 77 input artifacts. These shared-rate/cell comparisons are
descriptive, not independent statistical evidence, and establish no test
superiority. They help separate magnitude-rate effects from differences in
base-rate grid coverage.

The [independent gradient and optimizer tests](test_adapters.py) verify these
implementations. The [teacher generator](teacher.py) fixes one problem per
cell; seeds 42/43/44 vary adapter initialization. White/rescaled pairs preserve
the function and labels. The
[published teacher summary and audit](../../results/2026-10-08-round2/teacher_summary.json)
check 504 selected checkpoints; the smallest final normalized raw column norm
is 0.009955, far above the clamp. The [portable CPU verifier](verify_teacher_compact.py)
regenerates all 308 problem tensors with exact hashes and independently
reconstructs dense weights from the archived adapters. The results show rank-2
heterogeneous full-rank white teachers at relative test MSE 0.2721
for the combination, 0.00002336 with gains, and 0.00000001526 for standard DoRA.
Standard DoRA is already effective on these constructed teachers. They do not
establish a downstream winner or isolate representational limits from every
possible optimization failure.

**Arithmetic identity does not guarantee identical mixed-precision outputs.**
The [Aircraft initialization controls](../../results/2026-10-08-round2/numerical_controls.md)
use the trained baseline classifier as a diagnostic probe, rather than replaying
each adapter run's initial random classifier. Across all 3,333 validation images,
FP32 zero-update adapters change no argmax decisions. Under BF16, LoRA/NoRA and
the four DoRA-based methods form two internally bitwise-equal families whose
predictions differ on 43 images. Their numbers correct differ by only three
images; neither count is a correction or an error bound for trained results.
The four DoRA-based methods share the same initial numerical behavior, so this
cross-family discrepancy alone does not explain gains or slower-magnitude
results relative to the original combination.

CPU-initialized magnitudes divided by GPU-computed norms differ from one by up
to $2.384\times10^{-7}$. In a separate fresh rank-8 DoRA instance, recomputing
magnitudes with the GPU denominator path makes the ratio exactly one and the
first 128 validation-image logits bitwise equal to LoRA in FP32 and BF16.
This identifies a mechanism for the observed discrepancy in that probe. It does
not establish bitwise equality to native Linear, whose projection/bias execution
path still differs, or measure the effect on subsequent optimization.

For future controlled runs, initialize magnitudes using the denominator path on
the final device and dtype, before creating the optimizer. An explicit
fresh-adapter finalization step can preserve the original random factor draws
while doing this. It must never silently reset learned magnitudes on device
movement or checkpoint loading; a zero output factor alone does not prove that
an adapter is untrained. Verify the initialization after CPU-to-GPU transfer in
the intended precision. This remedy has only the zero-update probe evidence
above; the frozen benchmark comparisons were not modified or retrained.

Two **untested hypotheses** could isolate mechanisms in future work: initialize
gains to the original column norms to match ordinary DoRA's effective down
factor and initial output-factor gradient scale; or penalize log-gain changes
around their initialization to test whether amplitude flexibility needs
regularization on small datasets. Neither variant was run in this round.
