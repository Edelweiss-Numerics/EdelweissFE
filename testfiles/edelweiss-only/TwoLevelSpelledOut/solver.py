# A two-level cycle spelled out from general components, accelerated by LGMRES: smooth, correct on a coarse space,
# smooth again. The coarse space is the linear (P1) interpolant on the quadratic elements: the displacements of the
# corner nodes.
import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.composition import (
    AMG,
    LGMRES,
    CoarseCorrection,
    FixedTolerance,
    Jacobi,
    Multiplicative,
)


def p1Interpolation(A, context):
    """P, the prolongation from the corner nodes to all nodes: a corner node keeps its own displacement, a midside node
    takes the mean of the two corner nodes of its edge."""
    field = context.onlyField("p1Interpolation")
    isCorner, edgeEndpoints = context.p1Map(field.name)
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
    return sp.kron(nodalP, sp.identity(field.dimension), format="csr")


# the coarse problem, A_c = Pᵀ A P, solved approximately by one AMG cycle
coarseCorrection = CoarseCorrection(p1Interpolation, AMG())

# damped Jacobi before and after: one smoother, applied twice. It damps the fast modes only if omega < 2 / λ_max(D⁻¹A):
# on these quadratic elements λ_max is about 3.7, and the textbook omega = 2/3 would amplify them
smoother = Jacobi(omega=0.4)
twoLevelCycle = Multiplicative([smoother, coarseCorrection, smoother])

linearSolver = LGMRES(twoLevelCycle, tolerance=FixedTolerance(1e-8))
