# A two-level cycle spelled out from general components, accelerated by LGMRES: smooth, correct on a coarse space,
# smooth again. The coarse space is the linear (P1) interpolant on the quadratic elements: the displacements of the
# corner nodes.
import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.composition import (
    LGMRES,
    CoarseCorrection,
    DiagonalScaling,
    FixedTolerance,
    Jacobi,
    Multiplicative,
    RigidBodyModesPerAggregate,
    TwoLevel,
)


def cornerNodeInterpolation(A, context):
    """P, the prolongation from the corner nodes to all nodes: a corner node keeps its own displacement, a midside node
    takes the mean of the two corner nodes of its edge; and what the coarse dofs are: the displacement field on the
    corner nodes.

    The matrix here is not scaled (no DiagonalScaling above), so its variables are the displacements themselves; on a
    scaled matrix D A D, P would interpolate D⁻¹ u and take the weights of D, as CornerNodeInterpolation does."""
    field = context.onlyField("cornerNodeInterpolation")
    isCorner, edgeEndpoints = context.midsideCorners(field.name)
    cornerNumber = np.cumsum(isCorner) - 1  # the number of a corner node among the corner nodes
    rows, columns, weights = [], [], []
    for node in range(len(isCorner)):
        corners = [node] if isCorner[node] else edgeEndpoints[node]
        for corner in corners:
            rows.append(node)
            columns.append(cornerNumber[corner])
            weights.append(1.0 / len(corners))
    nodalP = sp.csr_matrix((weights, (rows, columns)), shape=(len(isCorner), np.count_nonzero(isCorner)))
    # each component of the displacement is interpolated alike; the dofs are numbered node by node
    P = sp.kron(nodalP, sp.identity(field.dimension), format="csr")
    # P keeps the Dirichlet rows (the native TwoLevel cycle decouples them instead): the coarse solve then treats a
    # Dirichlet corner dof as free, which is still a valid preconditioner. Zeroing the Dirichlet columns of P would
    # leave empty rows in A_c = Pᵀ A P, which the DiagonalScaling of the coarse solve cannot scale.
    # the coarse dofs are the displacements of the corner nodes: a displacement field of its own, with their coordinates
    return P, context.ofCoarseNodes(field, isCorner)


# the coarse problem, A_c = Pᵀ A P, is the displacement field on the corner nodes, so a solver for a displacement field
# solves it: scaled to a unit diagonal, one cycle of the rigid-body coarse level
coarseCorrection = CoarseCorrection(
    cornerNodeInterpolation, DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))
)

# damped Jacobi before and after: one smoother, applied twice. It damps the fast modes only if omega < 2 / λ_max(D⁻¹A):
# on these quadratic elements λ_max is about 3.7, and the textbook omega = 2/3 would amplify them
smoother = Jacobi(omega=0.4)
twoLevelCycle = Multiplicative([smoother, coarseCorrection, smoother])

linearSolver = LGMRES(twoLevelCycle, tolerance=FixedTolerance(1e-8))
