# Deep-learning notes

Short reference notes on techniques that come up in design discussions.

## MLB: low-rank bilinear pooling (Hadamard product)

*Kim et al. 2016, "Hadamard Product for Low-rank Bilinear Pooling" (MLB, ICLR 2017).*

- **Problem.** Combine two vectors x (e.g. an image feature) and y (e.g. a question feature) so every
  feature of x can interact with every feature of y. The full answer is a *bilinear* map:
  out_k = x^T W_k y, i.e. the outer product x ⊗ y followed by a linear layer. With n-dim inputs and o
  outputs that is n·n·o parameters, and the n² outer-product tensor must be built.
- **MLB.** Factor each W_k as low rank: out = P^T (U^T x ⊙ V^T y). Project both inputs to width d,
  multiply element-wise, project to the output. Cost and parameters are linear in d, not n².
  This is a rank-d (CP) approximation of the full bilinear map; in the paper it matched compact
  bilinear pooling (MCB, a sketched outer product) with fewer parameters.
- **Take-away.** The element-wise product of two projections is a standard, well-behaved way to get
  multiplicative interactions; it is not known to train badly. Usual extras: a nonlinearity (tanh) on
  each projection, and a residual/normalisation around it.
- **Related.** Gated units (GLU / SwiGLU: a ⊙ σ(b)) are the same idea inside one vector. MFB/MUTAN
  (2017) generalise MLB: sum-pooling over k factors, or a Tucker decomposition.

## Outer product vs element-wise product (the Pairformer write-back, K152-P)

- Outer product a_i ⊗ b_j (c_o each) then Linear(c_o² → c_z): c_o² interaction terms through a
  c_o bottleneck (a Tucker-style form). Element-wise a_i ⊙ b_j (width m) then Linear(m → c_z): m terms
  (the MLB form).
- AlphaFold's outer product *mean* averages outer products over the MSA sequences, which yields a
  covariance (co-evolution signal). With a single spectrum there is nothing to average, so that
  reason does not carry over. AlphaFold 3's Pairformer has no single → pair write-back in the trunk.
- **Exact cheaper outer product.** W(a_i ⊗ b_j) = Σ_c a_ic (W_c b_j): compute W_c b_j once per peak
  (N · c_o² · c_z), then contract with a_i in one matmul (N² · c_o · c_z). Same weights and output,
  no N² · c_o² tensor. This is `pair_writeback_impl="factored"`.
