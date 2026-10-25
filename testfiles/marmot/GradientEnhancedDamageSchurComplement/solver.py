# LGMRES to 1e-10 around the Schur complement of the softening zone: the elements whose kappa exceeds 0.03, with one
# ring of neighbours, are condensed (solved exactly); the remaining dofs get blockamg's field preconditioners in a
# block Gauss-Seidel sweep over the diagonally scaled Schur complement.
from edelweissfe.config.markerlibrary import getMarkerClass
from edelweissfe.linsolve.composition import (
    LGMRES,
    DiagonalScaling,
    FixedTolerance,
    Multiplicative,
    OnField,
    SchurComplement,
)
from edelweissfe.linsolve.composition.presets import blockAMGFieldPreconditioner

softeningZone = getMarkerClass("fieldOutput").fromOptions({"fieldOutput": "kappa", "threshold": 0.03, "halo": 1})


def fieldCorrections(context):
    return [OnField(field.name, blockAMGFieldPreconditioner(field, context)) for field in context.fields]


blockGaussSeidel = Multiplicative(fieldCorrections)
linearSolver = LGMRES(
    SchurComplement(condense=[softeningZone], preconditionS=DiagonalScaling(blockGaussSeidel)),
    tolerance=FixedTolerance(1e-10),
)
