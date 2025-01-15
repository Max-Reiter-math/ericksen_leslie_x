"""
We import a namespace that is as small as possible.

Naming convention
1st char: L = linear or N = Nonlinear or FP = Fixed Point approximating nonlinear
2nd char: L2 = L2 inner product or h = mass-lumped inner product
3rd char: Further additions, i.e. P = Nodal projection, D = Decoupled
"""


# from sim.models.decoupled_fp_solver import decoupled_fp_solver
from sim.models.linear_projection_methods import LL2, LL2P, Lh, LhP
from sim.models.decoupled_fp_solver import FPhD, FPL2D
from sim.models.linear_projection_dg import lpdg, ldg
