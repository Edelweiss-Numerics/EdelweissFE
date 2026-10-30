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

from typing import ClassVar

from edelweissfe.utils.schema import OptionSchemaProvider


class KeywordBase(OptionSchemaProvider):
    """Base class for top-level ``.inp`` keywords, discovered via the registry's ``"keyword"``
    category (see ``config/registry.py``).

    A ``KeywordBase`` subclass carries :attr:`schema` -- the grammar's single source of truth,
    built from :func:`~edelweissfe.utils.schema.schemaField`,
    :func:`~edelweissfe.utils.schema.subKeywordField` and
    :func:`~edelweissfe.utils.schema.datalineField` -- plus its own spelling and description
    (:attr:`keywordName`/:attr:`keywordDescription`), consumed by
    :mod:`edelweissfe.utils.inputfileparser` for lexing/validation and by
    :mod:`edelweissfe.utils.schemasurface` for the rendered grammar surface. Construction from a
    parsed ``.inp`` definition happens in the per-category adapters
    (``abqmodelconstructor``/``inputfilehelpers``/``StepManager``), not on this class: a keyword's
    grammar and its construction are separate concerns.
    """

    #: Schema of this keyword's line options, dataline payload and ``>>`` sub-blocks, or ``None``.
    schema: ClassVar[type | None] = None

    #: The keyword as written in the ``.inp`` file, in display case (e.g. ``"elSet"``, not the registry
    #: key ``"elset"``). Single source of truth for its spelling; set by every concrete subclass.
    keywordName: ClassVar[str]

    #: The keyword's human-readable description, rendered by the grammar surface.
    keywordDescription: ClassVar[str]
