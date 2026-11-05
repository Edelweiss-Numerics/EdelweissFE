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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

"""The entities that contribute a block to the system matrix -- elements, constraints, and in EdelweissMeshfree cells
and particles -- and the layout of that block.

An entity knows its block only in its own numbering: row ``a`` and column ``b`` of the block are its ``a``-th and
``b``-th degree of freedom. Where these lie in the global system (the global DOF list of the entity, see
:class:`~edelweissfe.numerics.dofmanager.DofManager`), and how the blocks are stored and summed, is the business of the
system matrix assembly (:mod:`edelweissfe.numerics.systemmatrixassembly`).
"""

import numpy as np


def denseBlockLocalEntries(nDof: int) -> tuple[np.ndarray, np.ndarray]:
    """The local row and column of every value of a dense ``nDof x nDof`` block, stored column-major.

    Value ``k`` of the block lies in row ``k % nDof`` and column ``k // nDof``. This is the definition of the dense
    layout; the direct CSR assembly applies the same rule in C++, without materialising the pairs.

    Parameters
    ----------
    nDof
        The number of degrees of freedom of the entity.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The local rows and the local columns (int32).
    """

    localRows = np.tile(np.arange(nDof, dtype=np.intc), nDof)
    localColumns = np.repeat(np.arange(nDof, dtype=np.intc), nDof)
    return localRows, localColumns


def localEntriesOfSubBlocks(subBlocks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The local row and column of every value of a block made of dense sub-blocks, each stored row by row, one
    after the other: the layout of the constraints that couple only some of their degrees of freedom.

    Parameters
    ----------
    subBlocks
        For each sub-block, in the order of the values, its local rows and its local columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The local rows and the local columns (int32), one per value of the block.
    """

    localRows = [np.repeat(rows, len(columns)) for rows, columns in subBlocks]
    localColumns = [np.tile(columns, len(rows)) for rows, columns in subBlocks]
    return (
        np.concatenate([np.zeros(0, dtype=np.intc)] + localRows).astype(np.intc),
        np.concatenate([np.zeros(0, dtype=np.intc)] + localColumns).astype(np.intc),
    )


def countEntriesOfBlock(entity) -> int:
    """The number of values of the block of ``entity``: ``nDof**2`` for a dense block, else the number of entries it
    lists in its ``localEntriesOfBlock``. An assembly counts once per equation system and stores where each block
    begins and ends.

    Parameters
    ----------
    entity
        An element, a constraint, or any other entity that contributes a block to the system matrix.

    Returns
    -------
    int
        The number of values.
    """

    if entity.couplesAllItsDofs:
        return entity.nDof**2
    return len(entity.localEntriesOfBlock()[0])


class SystemMatrixContributorBase:
    """Base class for the entities that contribute a block to the system matrix.

    The block couples the entity's degrees of freedom with each other. By default it couples all of them: the block is
    the dense ``nDof x nDof`` matrix, stored column-major. An entity that couples only some of its degrees of freedom
    (a contact constraint, which couples every slave node with the master side but not the slave nodes with each
    other) sets :attr:`couplesAllItsDofs` to ``False`` and lists the local row and column of each value of its block in
    :meth:`localEntriesOfBlock`. The number of values of a block is always the length of that list
    (:func:`countEntriesOfBlock`).

    Parameters
    ----------
    nDof
        The number of degrees of freedom of the entity.
    """

    def __init__(self, nDof: int):
        self.nDof = nDof

    @property
    def couplesAllItsDofs(self) -> bool:
        """Whether the entity's block couples every one of its degrees of freedom with every other one.

        ``True`` (the default, every element) means the block is the dense ``nDof x nDof`` matrix, stored
        column-major. ``False`` means the entity lists the entries of its block itself in
        :meth:`localEntriesOfBlock`, and its block is the list of their values, in that order.

        The entity says which, because the number of entries cannot tell: a sparse pattern can happen to have
        exactly ``nDof**2`` entries. A rigid body contact with a single slave node in 3D couples the six degrees of
        freedom of the reference point with each other, the three of the node with each other, and the two groups
        both ways: 36 + 9 + 18 + 18 = 81 = 9², yet its entries are ordered block by block, not column by column.

        Returns
        -------
        bool
            ``True`` for a dense block, ``False`` for a list of entries of the entity's own.
        """

        return True

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The local row and column of every value of the entity's block, in the order of the values.

        Local means in the entity's own list of degrees of freedom: row ``a`` is the entity's ``a``-th degree of
        freedom. The default is the dense column-major block (:func:`denseBlockLocalEntries`); an entity that does not
        couple all its degrees of freedom (:attr:`couplesAllItsDofs` is ``False``) overrides it.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The local rows and the local columns (int32), one per value of the block.
        """

        return denseBlockLocalEntries(self.nDof)

    def shapeBlock(self, values: np.ndarray) -> np.ndarray:
        """The entity's view of the flat array of the values of its block, which it writes its tangent into.

        By default the dense block as a 2-D ``nDof x nDof`` array (column-major) if :attr:`couplesAllItsDofs`, else the
        flat array itself. An entity with a block of its own layout returns a structured view of it.

        Parameters
        ----------
        values
            The values of the block, one per entry of :meth:`localEntriesOfBlock`.

        Returns
        -------
        numpy.ndarray
            The view the entity writes into.
        """

        if self.couplesAllItsDofs:
            return values.reshape((self.nDof, self.nDof), order="F")
        return values
