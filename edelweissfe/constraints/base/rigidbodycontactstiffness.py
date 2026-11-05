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
"""
The sparse stiffness layout of contact against a rigid body.

A rigid body enters a contact constraint through the six degrees of freedom of its reference point
(RP): displacement and rotation. Every slave block -- a slave node, or all parent-face nodes of a slave
facet -- couples to the RP and to itself, but never to another slave block. The stiffness slice is
therefore made of

- one RP self-block :math:`K_{rp,rp}`, shared by all slave blocks, and
- per slave block :math:`b`: a self-block :math:`K_{ss}^b`, and the couplings :math:`K_{s,rp}^b` and
  :math:`K_{rp,s}^b`.
"""

import numpy as np

from edelweissfe.numerics.systemmatrixcontributor import localEntriesOfSubBlocks


def rigidBodyContactLocalEntries(
    rpLocalIndices: np.ndarray, slaveBlockLocalIndices: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The local row and column of every sparse stiffness entry, in the order :class:`RigidBodyContactStiffnessView`
    expects: the RP self-block, then per slave block its self-block, the slave-to-RP and the RP-to-slave coupling,
    each block row by row.

    Parameters
    ----------
    rpLocalIndices
        The local indices of the RP degrees of freedom in the constraint's DOF list.
    slaveBlockLocalIndices
        The local indices of the degrees of freedom of each slave block.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The local rows and the local columns (int32).
    """

    subBlocks = [(rpLocalIndices, rpLocalIndices)]
    for slaveIndices in slaveBlockLocalIndices:
        subBlocks += [(slaveIndices, slaveIndices), (slaveIndices, rpLocalIndices), (rpLocalIndices, slaveIndices)]
    return localEntriesOfSubBlocks(subBlocks)


class RigidBodyContactStiffnessView:
    """Structured 2-D views onto the sparse stiffness slice of a rigid body contact constraint.

    Parameters
    ----------
    flat_array
        The constraint's slice of the sparse stiffness value array.
    rpDofCount
        The number of RP degrees of freedom.
    slaveBlockSizes
        The number of degrees of freedom of each slave block.

    Attributes
    ----------
    K_rprp : numpy.ndarray
        The RP self-block, of shape (rpDofCount, rpDofCount), shared by all slave blocks.
    K_ss : list[numpy.ndarray]
        The self-block of each slave block, of shape (m, m).
    K_srp : list[numpy.ndarray]
        The slave-to-RP coupling of each slave block, of shape (m, rpDofCount).
    K_rps : list[numpy.ndarray]
        The RP-to-slave coupling of each slave block, of shape (rpDofCount, m).
    """

    def __init__(self, flat_array: np.ndarray, rpDofCount: int, slaveBlockSizes: list[int]):
        def nextBlock(nRows: int, nColumns: int) -> np.ndarray:
            nonlocal offset
            block = flat_array[offset : offset + nRows * nColumns].reshape((nRows, nColumns))
            offset += nRows * nColumns
            return block

        offset = 0
        self.K_rprp = nextBlock(rpDofCount, rpDofCount)
        self.K_ss = []
        self.K_srp = []
        self.K_rps = []
        for m in slaveBlockSizes:
            self.K_ss.append(nextBlock(m, m))
            self.K_srp.append(nextBlock(m, rpDofCount))
            self.K_rps.append(nextBlock(rpDofCount, m))
