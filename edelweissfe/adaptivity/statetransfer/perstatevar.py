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

"""Composite state transfer routing individual named state variables to different strategies."""

import numpy as np

from edelweissfe.adaptivity.statetransfer.base import (
    StateTransferStrategy,
    perQuadraturePointBlockSize,
    quadraturePointReferenceCoordinates,
)


class PerStateVarStateTransfer(StateTransferStrategy):
    """Apply a *default* strategy to the whole per-quadrature-point state block, then override
    selected named state variables with their own strategies.

    This is the "project some, copy others" composite. It is deliberately policy-free: which state
    variables may be projected, copied, or reset depends entirely on the constitutive model, so the
    caller names the variables to treat specially and everything else follows the default. Each named
    override is located within a quadrature-point block via the element's ``getStateVarSlice(name)``
    hook, driven by the material's / element's own state-variable names.
    """

    def __init__(self, default: StateTransferStrategy, overrides: dict):
        self._default = default
        self._overrides = overrides  # {stateVarName: StateTransferStrategy}

    def transferState(self, parent, children):
        parentBlock = perQuadraturePointBlockSize(parent)
        parentValues = parent.getStateVars().reshape(parent.getNumberOfQuadraturePoints(), parentBlock)
        parentNodeCoords = np.array([n.coordinates for n in parent.nodes], dtype=float)
        parentRefCoords = quadraturePointReferenceCoordinates(parent, parentNodeCoords)

        # resolve each overridden name to its column range within one QP block
        overridden = np.zeros(parentBlock, dtype=bool)
        routed = []  # list of (strategy, columns)
        for name, strategy in self._overrides.items():
            offset, size = parent.getStateVarSlice(name)
            cols = np.arange(offset, offset + size)
            routed.append((strategy, cols))
            overridden[cols] = True
        defaultColumns = np.flatnonzero(~overridden)

        for child in children:
            childBlock = perQuadraturePointBlockSize(child)
            if childBlock != parentBlock:
                raise ValueError("state transfer: parent and child per-QP state sizes differ.")
            childInit = child.getStateVars().reshape(child.getNumberOfQuadraturePoints(), childBlock)
            childRefCoords = quadraturePointReferenceCoordinates(child, parentNodeCoords)

            result = childInit.copy()
            if defaultColumns.size:
                result[:, defaultColumns] = self._default._transferColumns(
                    parentValues,
                    parentRefCoords,
                    childRefCoords,
                    childInit,
                    defaultColumns,
                )
            for strategy, cols in routed:
                result[:, cols] = strategy._transferColumns(
                    parentValues, parentRefCoords, childRefCoords, childInit, cols
                )
            child.setStateVars(result.reshape(-1))

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        # the composite overrides transferState directly; the primitive is not used
        raise NotImplementedError("PerStateVarStateTransfer routes columns in transferState().")
