"""Offline *feasibility* prototype of nested block-AMG (B-AMG) on the dumped coupled fracture system.

Not a production solver, and not benchmarked on wall time: pyamg is pure-Python and serial (it even
re-enables the GIL), so its timings mean nothing against a 16-thread PARDISO. The one metric that
transfers to a parallel implementation (AMGCL, Trilinos/MueLu) is the **GMRES iteration count**,
which is a property of preconditioner quality, not of the arithmetic backend. This script measures
that count.

The question it answers: PERF_LINSOLVE_INVESTIGATION.md section 11 showed monolithic AMG stalls at
0.2-0.65 residual regardless of the near null-space. The literature (Alkmim et al. 2026) says the fix
is a *block* preconditioner -- AMG per field inside a block Gauss-Seidel sweep. Does that converge on
this model at all? pyamg (build-once / apply-many, native near-null-space) is the cheapest way to find
out, on the first dumped system.

Field split: displacement [0, dispDofs) (3 components, node-major) + nonlocal damage [dispDofs, n).
Displacement block gets the 3 rigid-body *translations* as near null-space (constructible from the DOF
layout alone -- no nodal coordinates); damage block gets the default constant.

Measured (first dumped system, 280,155 dof; direct PARDISO target ~11.5 s):
  - block-GS + per-field AMG CONVERGES: ~93-117 GMRES iterations to 1e-4, deviation ~1e-4..1e-6 from
    the direct solve -- where monolithic AMG never gets below 0.2. Block structure is the missing
    ingredient, exactly as the literature says.
  - The count is bottlenecked by the per-field AMG *quality*, not the block coupling (extra block-GS
    sweeps do not help; the displacement block alone needs ~136-175 iters with default pyamg SA, only
    3 levels for 214k dof -- an aggressive, weak hierarchy on this condensed elasticity + contact +
    tie operator). Bringing that down is the step-3 production job: a properly configured parallel AMG
    (MueLu/Trilinos as the paper uses, or a tuned AMGCL), the full 6 rigid body modes (rotations need
    nodal coordinates), Chebyshev/ILU smoothers, and a real strength measure.

Requires `pyamg` (not a project dependency): `pip install pyamg`. Run from a directory holding a
`linsolveDumps/` capture (see PERF_LINSOLVE_INVESTIGATION.md section 6).

Usage: python block_amg_prototype.py <dumpDir> [dispDofs] [rtol] [sweeps] [scale on|off]
"""

import sys
from time import perf_counter

import numpy as np
import pyamg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

dumpDir = sys.argv[1] if len(sys.argv) > 1 else "linsolveDumps"
dispDofs = int(sys.argv[2]) if len(sys.argv) > 2 else 214659
rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
sweeps = int(sys.argv[4]) if len(sys.argv) > 4 else 1  # inner block-GS sweeps

A = sp.load_npz(f"{dumpDir}/A_00_00002.npz").tocsr()
b = np.load(f"{dumpDir}/b_00_00002.npy")
n = A.shape[0]
d = dispDofs
print(f"system: {n} dof, {A.nnz} nnz;  displacement block {d}, damage block {n - d}", flush=True)

# Reference direct solve (target to beat), and reference solution for the true error.
pardiso = PardisoSolver(reuseSymbolicFactorization=True)
pardiso(A, b)  # absorb the analyze
t = perf_counter()
xref = pardiso(A, b)
directTime = perf_counter() - t
print(f"direct (PARDISO) solve: {directTime:.1f} s  (target)\n", flush=True)
bNorm = max(np.linalg.norm(b), 1e-300)
refNorm = max(np.linalg.norm(xref), 1e-300)

# Symmetric diagonal (Jacobi) scaling: A_hat = D^-1/2 A D^-1/2 has unit diagonal, which fixes the
# ~1e8 dynamic range (Dirichlet penalties + stiffness) that otherwise makes AMG's strength-of-
# connection and aggregation meaningless. Solve the scaled system, unscale the solution: with
# y = D^1/2 x, A_hat y = D^-1/2 b, and x = D^-1/2 y.
scale = sys.argv[5] if len(sys.argv) > 5 else "on"
if scale == "on":
    dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
    Dinv = sp.diags(dinv)
    A = (Dinv @ A @ Dinv).tocsr()
    b = b * dinv
    xref = xref / dinv  # y_ref = D^1/2 x_ref, for the deviation check in scaled space
    bNorm = max(np.linalg.norm(b), 1e-300)
    refNorm = max(np.linalg.norm(xref), 1e-300)
    print(
        "symmetric diagonal scaling ON: scaled diagonal in [{:.2e}, {:.2e}]\n".format(
            A.diagonal().min(), A.diagonal().max()
        ),
        flush=True,
    )

# Field blocks (contiguous, so plain slices).
Add = A[:d, :d].tocsr()
Adn = A[:d, d:].tocsr()
And = A[d:, :d].tocsr()
Ann = A[d:, d:].tocsr()

# Rigid-body translations for the displacement block: 1 on each component, from the DOF layout alone.
Bdisp = np.zeros((d, 3))
Bdisp[np.arange(d), np.arange(d) % 3] = 1.0
# Under symmetric scaling the near null-space transforms as B -> D^1/2 B (so that A_hat (D^1/2 v) =
# D^-1/2 A v ~ 0). Without this the AMG is handed the wrong near-null-space on the scaled operator.
if scale == "on":
    # Divide by the *pre-equilibration* scaling, matching what the production path does
    # (edelweissfe.linsolve.nullspace: `B / blockDinv[:, None]`). Scaling by Add.diagonal() here
    # would be a near no-op: Add is sliced from the already-equilibrated A, whose diagonal is ~1,
    # so the basis would stay effectively unscaled and this prototype would benchmark a different
    # near-null-space than production.
    Bdisp = Bdisp / dinv[:d][:, None]


def buildAMG(label, block, B):
    t = perf_counter()
    ml = pyamg.smoothed_aggregation_solver(block, B=B, max_coarse=500, keep=False)
    print(f"  {label}: {ml.levels[0].A.shape[0]} -> ... {len(ml.levels)} levels, built in {perf_counter() - t:.1f} s")
    return ml


print("building per-field AMG hierarchies:", flush=True)
mlDisp = buildAMG("displacement (SA, 3 translations)", Add, Bdisp)
mlDamage = buildAMG("damage (SA, constant)", Ann, None)
Pd = mlDisp.aspreconditioner(cycle="V")
Pn = mlDamage.aspreconditioner(cycle="V")
print(flush=True)


def makeBlockGS(nSweeps, symmetric):
    """Block Gauss-Seidel preconditioner apply. Each block is solved by one AMG V-cycle. `symmetric`
    adds a backward (damage-first) sweep, which folds *both* couplings in and usually halves the
    outer GMRES count on a coupled system."""

    def apply(r):
        rd, rn = r[:d], r[d:]
        xd = np.zeros(d)
        xn = np.zeros(n - d)
        for _ in range(nSweeps):
            xd = Pd(rd - Adn.dot(xn))
            xn = Pn(rn - And.dot(xd))
            if symmetric:
                xn = Pn(rn - And.dot(xd))
                xd = Pd(rd - Adn.dot(xn))
        return np.concatenate([xd, xn])

    return LinearOperator((n, n), matvec=apply, dtype=A.dtype)


# Diagnostic: how well does each field's AMG solve its OWN block? This localises the bottleneck --
# if the displacement block alone needs ~100 iters, the elasticity AMG (missing rotations) is the
# limit, not the block coupling.
for label, block, P in [("displacement A_dd alone", Add, Pd), ("damage A_nn alone", Ann, Pn)]:
    m = block.shape[0]
    rhs = np.ones(m)
    hist = []
    _, info = gmres(
        block,
        rhs,
        M=P,
        rtol=1e-4,
        atol=0.0,
        restart=100,
        maxiter=10,
        callback=lambda rr: hist.append(rr),
        callback_type="pr_norm",
    )
    print("  {:<28} {:>4} iters ({:})".format(label, len(hist), "conv" if info == 0 else "FAIL"), flush=True)
print()

print("{:<34} {:>6} {:>8} {:>12} {:>12}".format("preconditioner", "iters", "info", "true rel.res", "dev"))
for label, nSweeps, symmetric in [
    ("block-GS x1 (translations)", 1, False),
]:
    M = makeBlockGS(nSweeps, symmetric)
    history = []
    x, info = gmres(
        A,
        b,
        M=M,
        rtol=rtol,
        atol=0.0,
        restart=100,
        maxiter=10,
        callback=lambda rr: history.append(rr),
        callback_type="pr_norm",
    )
    trueRes = np.linalg.norm(A @ x - b) / bNorm
    dev = np.linalg.norm(x - xref) / refNorm
    print(
        "{:<34} {:>6} {:>8} {:>12.2e} {:>12.2e}".format(
            label, len(history), "conv" if info == 0 else "FAIL", trueRes, dev
        ),
        flush=True,
    )
print(f"\n  vs direct {directTime:.1f} s;  monolithic AMG stalled ~2e-1 at 300 iters", flush=True)
