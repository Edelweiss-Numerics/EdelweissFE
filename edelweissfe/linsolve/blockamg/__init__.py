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
    """Create a field-split block-AMG linear solver from a ``linsolverConfigFile``'s options.

    The factory the ``linsolver`` registry category resolves for the name ``blockamg``. No field
    structure is configured here: it is discovered from the model and pushed in by the nonlinear
    solver (see :class:`~edelweissfe.linsolve.base.LinearSolver`).

    Parameters
    ----------
    opts
        The options, one per keyword argument of
        :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` (documented there, and in the key
        tables of the theory page), each converted from its JSON value by
        :data:`~edelweissfe.linsolve.blockamg.blockamg.CONFIG_CASTS`. A key that is not one of them raises a
        ``ValueError`` instead of being ignored, and so does a value that cannot be converted. A non-mapping
        ``opts`` is tolerated (the implicit-static solver passes ``""`` when no configuration file is given),
        in which case every default applies.

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


class BlockAMGType:
    """``blockamg`` as a type of linear solver (registry category ``linsolver``): :class:`BlockAMGSolver` with the
    options of a ``linsolverConfigFile``, see :func:`createSolver`."""

    schema = None

    @classmethod
    def fromOptions(cls, options: dict) -> Callable:
        return createSolver(options)
