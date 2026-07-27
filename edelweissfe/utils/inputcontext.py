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

"""``InputContext``: a plain parameter object for the L4 adapter layer (see
``PLAN_INPUT_SYSTEM.md``, P1).

Today, an L4-shaped helper such as ``createFieldOutputFromInputFile(inputfile, model, journal)``
(``edelweissfe/helpers/inputfilehelpers.py:71``) and the ``outputManagerFactory(name, FEModel,
fieldOutputController, moduleOptions, journal, plotter, **kwargs)`` shape (finding #6 of the plan)
both thread the same handful of collaborators -- the model, the journal, the plotter, the
field-output controller -- through call frame after call frame as positional parameters. This is
the "``moduleOptions`` disease": every new construction helper added along the way has to repeat
the same parameter list, in the same order, or reach for a global instead.

``InputContext`` replaces the repetition with a single typed, immutable value that is constructed
once (by whatever assembles a simulation -- ``inputfileparser.py`` today, an EdelweissMeshfree
script tomorrow) and passed down by reference.

It is deliberately **not** a service locator: it has no lookup methods, performs no registry
access, and is never mutated after construction (``frozen=True``). It is exactly as "smart" as a
plain 4-tuple with names -- nothing more. In particular, and per this codebase's explicit
convention (a ``Journal`` is never a singleton or a global -- see
``edelweissfe/journal/journal.py``), the ``journal`` field is an ordinary attribute supplied by
the caller at construction time, not something ``InputContext`` reaches for on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter


@dataclass(frozen=True)
class InputContext:
    """Carries the collaborators an L4 adapter needs to construct L1 objects.

    Parameters
    ----------
    model
        The :class:`~edelweissfe.models.femodel.FEModel` being assembled or simulated.
    journal
        The :class:`~edelweissfe.journal.journal.Journal` used for logging. Always supplied
        explicitly by the caller -- never a global/singleton (see module docstring).
    plotter
        The :class:`~edelweissfe.utils.plotter.Plotter` used for interactive/diagnostic plots, if
        any is in use for this run.
    fieldOutputController
        The :class:`~edelweissfe.utils.fieldoutput.FieldOutputController` collecting this run's
        field outputs, if one has been created yet.
    """

    model: FEModel
    journal: Journal
    plotter: Plotter | None = None
    fieldOutputController: FieldOutputController | None = None
