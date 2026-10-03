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

"""Pluggable quadrature-point state-variable transfer for adaptive mesh refinement (WS-F).

When a parent element is subdivided, the material history stored at its quadrature points must be
handed down to the freshly created children. Different internal variables call for different
strategies -- an already-admissible tensor may simply be copied from the nearest parent point, a
smoothly-varying field may be projected, a rate/flag variable may be reset to its virgin value --
so the transfer is expressed as an interchangeable :class:`StateTransferStrategy` rather than a
single hard-coded rule. A :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer`
composite routes individual named state variables to different strategies (e.g. *project* the
strain but *copy* everything else).

Every strategy is built on one primitive, :meth:`StateTransferStrategy._transferColumns`, which
maps a subset of the per-quadrature-point state block from parent to child. The concrete element
must expose its flat state buffer via ``getStateVars`` / ``setStateVars`` (laid out as
``nQuadraturePoints`` equal contiguous blocks); per-name routing additionally requires
``getStateVarSlice(name)`` to locate a named variable within one block.
"""

from abc import ABC, abstractmethod

import numpy as np


def perQuadraturePointBlockSize(element) -> int:
    """Number of state-variable doubles per quadrature point (the flat buffer is
    ``nQuadraturePoints`` such blocks)."""
    n = element.getStateVars().shape[0]
    nqp = element.getNumberOfQuadraturePoints()
    if nqp == 0 or n % nqp != 0:
        raise ValueError(
            "state transfer: state buffer of size {:} is not an integer multiple of {:} "
            "quadrature points (element-level state variables are not supported).".format(n, nqp)
        )
    return n // nqp


def quadraturePointReferenceCoordinates(element, referenceNodeCoords, topology) -> np.ndarray:
    """Reference-cube coordinates (in the frame of ``referenceNodeCoords``) of ``element``'s
    quadrature points. Passing the parent's node coordinates maps both the parent's and the
    children's quadrature points into the same (parent) reference frame, which is what makes
    nearest-point matching and projection distortion-independent and octant-correct."""
    phys = np.asarray(element.getCoordinatesAtQuadraturePoints()).reshape(element.getNumberOfQuadraturePoints(), -1)
    return np.array([topology.inverse_map(p, referenceNodeCoords) for p in phys])


class StateTransferStrategy(ABC):
    """Interchangeable strategy for transferring quadrature-point material state from a refined
    parent element to its children.

    Subclasses implement the single primitive :meth:`_transferColumns`; the shared driver
    :meth:`transferState` handles buffer (re)shaping, the parent/child reference-frame mapping and
    writing the result back. A strategy is therefore agnostic of the HEX20 machinery and only needs
    to answer "given the parent's per-quadrature-point values, what are the children's?".
    """

    def transferState(self, parent, children, topology):
        """Transfer the whole state block of ``parent`` into each element of ``children``.

        Parameters
        ----------
        parent
            The (converged) parent element being refined.
        children
            The freshly created & initialised child elements (same type / material as the parent).
        topology
            The TopologyBase instance for the element type.
        """
        parentBlock = perQuadraturePointBlockSize(parent)
        parentValues = parent.getStateVars().reshape(parent.getNumberOfQuadraturePoints(), parentBlock)
        parentNodeCoords = np.array([n.coordinates for n in parent.nodes], dtype=float)
        parentRefCoords = quadraturePointReferenceCoordinates(parent, parentNodeCoords, topology)

        allColumns = np.arange(parentBlock)
        for child in children:
            childBlock = perQuadraturePointBlockSize(child)
            if childBlock != parentBlock:
                raise ValueError("state transfer: parent and child per-QP state sizes differ.")
            childInit = child.getStateVars().reshape(child.getNumberOfQuadraturePoints(), childBlock)
            childRefCoords = quadraturePointReferenceCoordinates(child, parentNodeCoords, topology)

            result = childInit.copy()
            result[:, allColumns] = self._transferColumns(
                parentValues, parentRefCoords, childRefCoords, childInit, allColumns
            )
            child.setStateVars(result.reshape(-1))

    @abstractmethod
    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns) -> np.ndarray:
        """Compute the child's values for a subset of the per-quadrature-point state columns.

        Parameters
        ----------
        parentValues
            ``(nQpParent, blockSize)`` parent per-quadrature-point state.
        parentRefCoords
            ``(nQpParent, 3)`` parent quadrature-point coordinates in the parent reference cube.
        childRefCoords
            ``(nQpChild, 3)`` child quadrature-point coordinates in the *parent* reference cube.
        childInitValues
            ``(nQpChild, blockSize)`` the child's freshly-initialised state (virgin values).
        columns
            Index array selecting the state columns to transfer.

        Returns
        -------
        np.ndarray
            ``(nQpChild, len(columns))`` the child's values for the selected columns.
        """
