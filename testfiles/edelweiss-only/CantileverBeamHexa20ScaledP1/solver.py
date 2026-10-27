# Three levels for a field of quadratic hexahedra, composed from building blocks: the quadratic field smoothed by a
# Chebyshev polynomial, its linear (P1) part on the corner nodes as the coarse level, and on that the rigid-body modes
# of aggregates of corner nodes.
from edelweissfe.linsolve.composition import (
    LGMRES,
    DiagonalScaling,
    FixedTolerance,
    OnField,
    P1Interpolation,
    RigidBodyModes,
    TwoLevel,
)

# the coarse problem is the displacement field on the corner nodes: scaled to a unit diagonal once more, and solved by
# the rigid-body coarse level (this is P1Interpolation's default coarse solve, written out)
coarse = DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModes()))
displacement = OnField("displacement", TwoLevel(coarseSpace=P1Interpolation(coarse=coarse)))

linearSolver = LGMRES(DiagonalScaling(displacement), tolerance=FixedTolerance(1e-8))
