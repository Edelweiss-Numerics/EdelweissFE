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

from dataclasses import dataclass

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
    other) lists the local row and column of each value of its block: it overrides :meth:`localEntriesOfBlock`, and
    that is all it does to declare the layout of its block (:attr:`couplesAllItsDofs` follows from it). The number of
    values of a block is always the length of that list (:func:`countEntriesOfBlock`).

    The layout is declared once, by the override: a subclass that declares :attr:`couplesAllItsDofs` itself is refused
    when it is defined, as a second declaration could contradict the first. An extension type, which cannot inherit
    this class (the Marmot elements, cells and particles), declares both itself: a dense block, whose
    :meth:`localEntriesOfBlock` is :func:`denseBlockLocalEntries`.

    Parameters
    ----------
    nDof
        The number of degrees of freedom of the entity.
    """

    def __init_subclass__(cls, **kwargs):
        """Refuse a subclass that declares :attr:`couplesAllItsDofs` itself: the layout of its block follows from
        whether it overrides :meth:`localEntriesOfBlock`.

        Raises
        ------
        TypeError
            If the subclass declares :attr:`couplesAllItsDofs`.
        """

        super().__init_subclass__(**kwargs)
        if cls.couplesAllItsDofs is not SystemMatrixContributorBase.couplesAllItsDofs:
            raise TypeError(
                "{:} declares couplesAllItsDofs: the layout of the block follows from localEntriesOfBlock alone, "
                "which an entity that couples only some of its degrees of freedom overrides.".format(cls.__qualname__)
            )

    def __init__(self, nDof: int):
        self.nDof = nDof

    @property
    def couplesAllItsDofs(self) -> bool:
        """Whether the entity's block couples every one of its degrees of freedom with every other one.

        ``True`` (the default, every element) means the block is the dense ``nDof x nDof`` matrix, stored
        column-major. ``False`` means the entity lists the entries of its block itself, by overriding
        :meth:`localEntriesOfBlock`, and its block is the list of their values, in that order. It is ``False``
        exactly for the entities that override it.

        The layout is never guessed from the number of entries, which cannot tell: a sparse pattern can happen to
        have exactly ``nDof**2`` entries. A rigid body contact with a single slave node in 3D couples the six degrees
        of freedom of the reference point with each other, the three of the node with each other, and the two groups
        both ways: 36 + 9 + 18 + 18 = 81 = 9², yet its entries are ordered block by block, not column by column.

        Returns
        -------
        bool
            ``True`` for a dense block, ``False`` for a list of entries of the entity's own.
        """

        return type(self).localEntriesOfBlock is SystemMatrixContributorBase.localEntriesOfBlock

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The local row and column of every value of the entity's block, in the order of the values.

        Local means in the entity's own list of degrees of freedom: row ``a`` is the entity's ``a``-th degree of
        freedom. The default is the dense column-major block (:func:`denseBlockLocalEntries`); an entity that does not
        couple all its degrees of freedom overrides it, which makes its block the list of the values of these entries
        (:attr:`couplesAllItsDofs` is then ``False``).

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


@dataclass(frozen=True)
class SystemMatrixEntities:
    """The entities of an equation system as the direct CSR assembly reads them: flat arrays, the counterpart of the
    C++ struct of the same name in ``_csrcore.h``.

    Entity ``e`` has the global DOF list ``dofs[dofStart[e] : dofStart[e + 1]]``. A dense block (``couplesAllItsDofs``)
    is described by that list alone; the block of any other entity is the list of entries it gives in
    :meth:`SystemMatrixContributorBase.localEntriesOfBlock`, stored in ``localRows`` and ``localColumns`` from
    ``localEntryStart[e]`` to ``localEntryStart[e + 1]``. A dense entity lists no entries: its ``nDof**2`` pairs are
    never materialised. The lists of the entities lie one after the other: ``dofStart`` and ``localEntryStart`` begin
    at 0 and never decrease, and the direct assembly refuses a table whose starts do not.
    """

    #: where the DOF list of each entity begins in :attr:`dofs` (int64, one more than there are entities)
    dofStart: np.ndarray
    #: the global DOF lists of all entities, one after the other (int32)
    dofs: np.ndarray
    #: whether the block of each entity is dense (uint8)
    couplesAllItsDofs: np.ndarray
    #: where the listed entries of each entity begin in :attr:`localRows` and :attr:`localColumns` (int64, one more
    #: than there are entities)
    localEntryStart: np.ndarray
    #: the local row of every listed entry (int32)
    localRows: np.ndarray
    #: the local column of every listed entry (int32)
    localColumns: np.ndarray

    @classmethod
    def ofEntities(cls, dofsOfEntity: dict) -> "SystemMatrixEntities":
        """The table of the entities, in the order of ``dofsOfEntity``.

        Parameters
        ----------
        dofsOfEntity
            The global DOF list of every entity.

        Returns
        -------
        SystemMatrixEntities
            The table.
        """

        entities = list(dofsOfEntity)
        nEntities = len(entities)
        noEntries = np.zeros(0, dtype=np.intc)

        couplesAllItsDofs = np.fromiter((entity.couplesAllItsDofs for entity in entities), np.uint8, nEntities)
        listedEntries = [
            (noEntries, noEntries) if couplesAll else entity.localEntriesOfBlock()
            for entity, couplesAll in zip(entities, couplesAllItsDofs)
        ]

        def startsOf(lengths) -> np.ndarray:
            starts = np.zeros(nEntities + 1, dtype=np.int64)
            np.cumsum(np.fromiter(lengths, np.int64, nEntities), out=starts[1:])
            return starts

        return cls(
            dofStart=startsOf(len(dofs) for dofs in dofsOfEntity.values()),
            dofs=np.concatenate([noEntries, *dofsOfEntity.values()]).astype(np.intc),
            couplesAllItsDofs=couplesAllItsDofs,
            localEntryStart=startsOf(len(localRows) for localRows, _ in listedEntries),
            localRows=np.concatenate([noEntries, *(localRows for localRows, _ in listedEntries)]).astype(np.intc),
            localColumns=np.concatenate([noEntries, *(columns for _, columns in listedEntries)]).astype(np.intc),
        )
