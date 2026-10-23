# Appendix E numerical example

[instance.mps](instance.mps) defines the author-created convex quadratic program in standard MPS with a `QUADOBJ` section:

Minimize `0.5 * x'Q*x + c'x`, with `Q = [[20, 0.1], [0.1, 0.2]]` and `c = [-1, -1]`, subject to `x1+x2 <= 5`, `3*x1+x2 <= 10`, and `x1,x2 >= 0`. The off-diagonal coefficient is stored once in `QUADOBJ`; the reader symmetrizes it.

Both HPR-QP and HPR-p1 support this format through their existing MPS readers. The appendix driver reads the model from this file; it does not hardcode the coefficients. The MPS file is distributed with the authors' MIT-licensed reproduction code.

From the repository root, run `./scripts/run.sh Appendix-Example HPR-QP` or `HPR-p1`. The dedicated driver preserves the paper's fixed spectral bounds, fixed sigma, disabled scaling and stopping-check interval. Those algorithm settings are defined directly in the [appendix test script](../../scripts/run_concrete_example.jl). Using an ordinary benchmark driver's default settings can produce different iteration counts.

Existing stopping-rule and historical iteration-count qualifications remain in [validation](../../docs/VALIDATION.md).
