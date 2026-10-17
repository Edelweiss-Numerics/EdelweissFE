#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

"""Registry-facing factory for the field-split block-AMG linear solver.

The implementation lives in :mod:`edelweissfe.linsolve.blockamg.blockamg`; see there for the method.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create a field-split block-AMG linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``blockamg``. No field
    structure is configured here: it is discovered from the model and pushed in by the nonlinear
    solver (see :class:`~edelweissfe.linsolve.base.LinearSolver`).

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. All optional (see
        :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`):

        ``outerTol``, ``outerRestart``, ``outerMaxiter``, ``sweeps``, ``symmetric``
            The outer GMRES and block Gauss-Seidel knobs. ``outerTol`` unset, ``None``, or the literal
            string ``"adaptive"`` (JSON has no bare ``null`` in this cast pipeline) all mean the same
            thing: use Eisenstat--Walker adaptive forcing (the default) instead of a fixed tolerance --
            pass an actual float to pin a fixed outer GMRES relative tolerance instead. See
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` for the forcing scheme
            itself.
        ``outerSolver``, ``lgmresM``, ``lgmresK``
            ``outerSolver`` selects the outer Krylov solve: ``"amgcl_lgmres"`` (default -- AMGCL's own
            native ``amgcl::solver::lgmres`` in place of ``scipy.sparse.linalg.gmres``, live-gated at
            several thread counts, faster than SciPy at every thread count tested and increasingly so
            as thread count grows) or ``"scipy"`` (the prior default, kept as a fallback/opt-out). The
            other two are only used with ``"amgcl_lgmres"``; see
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` for their meaning.
        ``lgmresAlwaysReset``, ``lgmresResetOnNewIncrement``
            Deprecated: LGMRES recycling across solves was removed. ``true`` / ``false`` respectively
            are still accepted, with a deprecation note; a value asking for recycling raises.
        ``etaMin``, ``etaMax``, ``ewGamma``, ``ewAlpha``, ``residualGrowthFactor``,
        ``hierarchyStalenessFactor``, ``refreshOnNewIncrement``
            Knobs for the adaptive outer tolerance and the per-field AMG hierarchy reuse across Newton
            iterations -- see :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.
        ``trueResidualMaxContinuations``
            How many warm-restart continuations enforce the requested tolerance on the true residual,
            not just GMRES's own preconditioned stopping check. Defaults to ``2``; ``0`` restores the
            original preconditioned-residual-only behaviour.
        ``verbosity``
            ``"silent"``, ``"warning"`` (default), ``"info"``, or ``"debug"`` -- see
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`. Replaces the old boolean
            ``verbose`` option entirely; existing configs setting ``verbose`` should switch to this.
        ``warnOuterIterationsThreshold``
            Outer-iteration count past which a solve prints a ``"warning"``-level message even at the
            default verbosity.
        ``dumpOnDegradationDir``, ``dumpOnDegradationThreshold``, ``dumpOnDegradationMaxDumps``,
        ``dumpOnDegradationContextSolves``
            Capture the raw ``(A, b)`` and field-block layout of solves that degrade (outer-iteration
            count past a threshold), plus optionally a window of preceding solves and per-solve
            solver-state bookkeeping, for offline diagnosis -- see
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`. ``dumpOnDegradationDir``
            unset (the default) disables this entirely.
        ``fieldPreconds``
            Optional mapping of field name (e.g. ``"displacement"``) to an AMGCL preconditioner
            parameter tree, overriding the dimension-based default for that field.
        ``useRigidBodyNullspace``
            ``True`` (default) builds a vector field's near null-space as the full rigid-body basis
            (translations + rotations) once nodal coordinates arrive, instead of translations alone --
            see :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`. Set ``False`` to force
            translations-only unconditionally.
        ``p1FieldNames``
            Optional list of vector field names (e.g. ``["displacement"]``) to precondition with
            p-multigrid instead of the single-level AMGCL default -- an opt-in variant, not the
            recommended configuration (see the module docstring of
            :mod:`edelweissfe.linsolve.blockamg.ptwogrid`). The actual topology map is computed
            lazily by the solver itself, from the live model reference ``setModel`` provides, the
            first time one of these fields' hierarchies is built -- not known at construction time,
            so nothing needs to be pushed in ahead of it.
        ``p1Nu``, ``p1NuPre``, ``p1NuPost``, ``p1FineDegree``, ``p1FineBlockSize``, ``p1FineValueType``,
        ``p1CoarseValueType``, ``p1FineSpectralRadiusReuse``, ``p1CoarsePrecond``
            Settings of the p-multigrid preconditioner of the ``p1FieldNames`` fields: fine Chebyshev
            sweeps (both, or pre and post separately) and degree, the fine operator's storage (``1`` scalar, ``3`` node-wise blocks) and
            value type (``"double"``, ``"float"``), the coarse hierarchy's value type,
            whether to reuse the fine spectral-radius estimate across rebuilds, and the coarse
            level's AMGCL tree -- see :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.

        ``hierarchyDropTol``, ``hierarchyDropLumping``
            Sparsification of each field's diagonal block before its AMG hierarchy is built -- see
            :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.
        ``gapMaxFactor``
            Upper bound of the gap factor ``gapCompensatedTolerance`` applies.

        A key not listed here raises a ``ValueError`` instead of being ignored.

        As with the other factories, a non-mapping ``opts`` is tolerated (the implicit-static solver
        passes ``""`` when no configuration file is given), in which case every default applies.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`, callable as ``(A, b) -> x``.
    """

    from edelweissfe.linsolve.blockamg.blockamg import (
        _DEPRECATED_LGMRES_KEYS,
        CONFIG_CASTS,
        BlockAMGSolver,
    )

    optionMap = opts if isinstance(opts, Mapping) else {}

    kwargs = {}
    for key, value in optionMap.items():
        if key in CONFIG_CASTS:
            try:
                kwargs[key] = CONFIG_CASTS[key](value)
            except (TypeError, ValueError) as error:
                raise ValueError("blockamg: configuration key {:}: {:}".format(key, error)) from error
    recognized = set(CONFIG_CASTS)

    # A key this factory does not know would otherwise be dropped without a word -- a typo, or a
    # setting that only the constructor takes, then silently has no effect on the run.
    unknown = sorted(set(optionMap) - recognized)
    if unknown:
        raise ValueError(
            "blockamg: unknown configuration key(s) {:}; recognized keys are {:}.".format(
                ", ".join(unknown), ", ".join(sorted(recognized))
            )
        )

    solver = BlockAMGSolver(**kwargs)
    deprecated = sorted(set(optionMap) & set(_DEPRECATED_LGMRES_KEYS))
    if deprecated:
        solver._log(
            "warning",
            "blockamg: {:} deprecated -- LGMRES recycling across solves was removed and these keys no "
            "longer have an effect; drop them from the configuration.".format(", ".join(deprecated)),
        )
    return solver
