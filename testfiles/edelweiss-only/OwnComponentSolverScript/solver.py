# A component of our own, composed with those of the library: a symmetric Gauss-Seidel smoother before and after one
# AMG cycle, accelerated by LGMRES.
from scipy.sparse import tril, triu
from scipy.sparse.linalg import spsolve_triangular

from edelweissfe.linsolve.composition import (
    AMG,
    LGMRES,
    Description,
    FixedTolerance,
    Multiplicative,
    Preconditioner,
)


class SymmetricGaussSeidel(Preconditioner):
    """One forward and one backward Gauss-Seidel sweep: z = (D + U)⁻¹ D (D + L)⁻¹ r, with A = L + D + U split into its
    strictly lower triangle, its diagonal and its strictly upper triangle."""

    def _forget(self):
        # what _setUp builds, before it is built (also called by the constructor of Preconditioner)
        self.lower = self.upper = self.diagonal = None

    def _setUp(self, A, context):
        # A is a SystemMatrix, as every component gets it; A.matrix() is the SciPy matrix
        matrix = A.matrix()
        self.lower = tril(matrix, format="csr")  # D + L
        self.upper = triu(matrix, format="csr")  # D + U
        self.diagonal = matrix.diagonal()  # D

    def _update(self, A, change):
        # only the values changed: the triangles of the last matrix still make a good smoother for the new one
        if change.requiresRebuild or self.lower is None:
            self.setUp(A, change.context)

    def _apply(self, r):
        y = spsolve_triangular(self.lower, r, lower=True)  # forward sweep: (D + L) y = r
        return spsolve_triangular(self.upper, self.diagonal * y, lower=False)  # backward sweep: (D + U) z = D y

    def describe(self):
        return Description(
            name="SymmetricGaussSeidel", formula="(D + U)⁻¹ D (D + L)⁻¹", latex="(D + U)^{-1} D (D + L)^{-1}"
        )


# smooth, one AMG cycle, smooth again: one smoother, applied twice
gaussSeidel = SymmetricGaussSeidel()
cycle = Multiplicative([gaussSeidel, AMG(), gaussSeidel])

linearSolver = LGMRES(cycle, tolerance=FixedTolerance(1e-8))
