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
"""The length check of a buffer that a Cython wrapper hands to C++ as a bare pointer.

A wrapper of a Marmot entity passes ``&buffer[0]``, and Marmot reads or writes as many entries as the
entity has, e.g. ``nDof`` for a residual or ``nDof * nDof`` for a stiffness. Marmot cannot check the
length of a bare pointer, and the extensions are compiled with ``boundscheck=False``, so a buffer that
is too short would be written past its end without any message. Every wrapper therefore checks each
buffer once, before it hands the pointer over::

    from edelweissfe.utils.bufferlength cimport requireLength

    requireLength("computeKernels", "Pe", Pe.shape[0], self._nDof)

The check is a single integer comparison; it is placed before a ``with nogil:`` block, as raising
needs the GIL.
"""


cdef inline int requireLength(str method, str buffer, Py_ssize_t length, Py_ssize_t requiredLength) except -1:
    """Raise a ValueError if a buffer does not have the length the C++ side reads or writes.

    Parameters
    ----------
    method
        The name of the wrapper method, for the message.
    buffer
        The name of the buffer (or of its axis), for the message.
    length
        The length the buffer has.
    requiredLength
        The length the C++ side reads or writes.
    """

    if length != requiredLength:
        raise ValueError(f"{method}: {buffer} has {length} entries, but {requiredLength} are required")
    return 0
