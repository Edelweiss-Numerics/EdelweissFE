# Three levels for a field of quadratic hexahedra, composed from building blocks: the quadratic field smoothed by a
# Chebyshev polynomial, its linear (P1) part on the corner nodes as the coarse level, and on that the rigid-body modes
# of aggregates of corner nodes.
from edelweissfe.linsolve.composition import (
    LGMRES,
    CornerNodeInterpolation,
    DiagonalScaling,
    FixedTolerance,
    OnField,
    RigidBodyModesPerAggregate,
    TwoLevel,
)
from edelweissfe.linsolve.composition.amg import defaultPTwoGridTree

# the coarse problem is the displacement field on the corner nodes: scaled to a unit diagonal once more, and solved by
# the rigid-body coarse level with the smoother of the corner-node two-level cycle (this is CornerNodeInterpolation's default coarse solve,
# written out)
coarse = DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate(), tree=defaultCornerNodeTwoLevelTree()))
displacement = OnField("displacement", TwoLevel(coarseSpace=CornerNodeInterpolation(coarseSolve=coarse)))

linearSolver = LGMRES(DiagonalScaling(displacement), tolerance=FixedTolerance(1e-8))
