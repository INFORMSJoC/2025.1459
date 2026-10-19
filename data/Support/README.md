# Support input (not a paper benchmark)

`warmup.mps` is an author-created, two-variable convex QP, added for portable initialization without redistributing third-party benchmark files. It minimizes `x1^2 + x2^2 - x1 - 2*x2` subject to `x1+x2=1` and `0<=x1,x2<=1`. Its solution is `(0.25, 0.75)`, objective `-1.125`.

This file is distributed under the repository MIT license and is not included in the 279 paper inputs or reported benchmark results. JuMP competitors and two historical dataset-driver helpers use it only for warmup. The unified HPR worker still warms up on the selected case.
