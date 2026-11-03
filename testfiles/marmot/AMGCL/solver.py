# The AMGCL test case without the standalone amgcl linear solver (deleted): GMRES (here AMGCL's LGMRES) to 1e-8,
# preconditioned by one smoothed-aggregation AMG hierarchy (AMGCL) on the displacement field.
from edelweissfe.linsolve.composition import AMG, LGMRES, FixedTolerance

linearSolver = LGMRES(AMG(), tolerance=FixedTolerance(1e-8))
