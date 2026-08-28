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

"""``*restart``: configures checkpoint/restart.

A new structural keyword, following the ``*job`` pattern (options only, no dataline payload,
see ``edelweissfe.keywords.job``) rather than EdelweissMeshfree's ``solveStep(...)`` kwarg pile --
EdelweissFE is keyword/schema-driven for everything else in its input language.

Writing a checkpoint (``write=True``) is wired through a dedicated output manager
(``edelweissfe.outputmanagers.restart``, registered as ``*output, type=restart``) that reuses this
keyword's parsed config, rather than duplicating ``writeInterval``/``baseName``/
``numberOfFilesToKeep`` on the output manager's own schema. Resuming (``readFrom=...``) is read
directly by the driver (``edelweissfe.drivers.inputfiledrivensimulation``) before the first step is
generated.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class RestartSchema:
    """L2: the options of the ``*restart`` keyword. No dataline payload -- see the module
    docstring."""

    write: bool = schemaField(description="write restart checkpoints during the analysis", dtype=bool, default=False)
    writeInterval: int = schemaField(
        description="write a checkpoint every N converged increments", dtype=int, default=1
    )
    baseName: str = schemaField(description="base file name for restart checkpoints", dtype=str, default="restart")
    numberOfFilesToKeep: int = schemaField(
        description="number of most recent restart checkpoints to keep (ring buffer)", dtype=int, default=3
    )
    readFrom: str | None = schemaField(
        description="path to an existing restart checkpoint to resume the analysis from",
        dtype=str,
        default=None,
    )


class RestartKeyword(KeywordBase):
    """``*restart``: configure checkpoint/restart."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = RestartSchema

    keywordName = "restart"
    keywordDescription = "configure checkpoint/restart"
