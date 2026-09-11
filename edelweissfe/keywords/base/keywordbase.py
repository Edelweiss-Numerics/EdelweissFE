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

    #: The schema dataclass describing this keyword's own line options, dataline payload, and
    #: ``>>`` sub-blocks, or ``None`` if it declares none. See
    #: :class:`~edelweissfe.utils.schema.OptionSchemaProvider`.
    schema: ClassVar[type | None] = None

    #: The keyword's identifier as written in the ``.inp`` file, in its exact display case (e.g.
    #: ``"elSet"``, ``"analyticalField"`` -- NOT the casefolded registry key ``"elset"``). This is
    #: the single source of truth for the keyword's spelling: the grammar surface renders it and
    #: the parser resolves it, so it must not be re-transcribed anywhere else. Declared (annotation
    #: only) on the base; every concrete subclass sets it.
    keywordName: ClassVar[str]

    #: The keyword's human-readable description. Rendered by the grammar surface; the single
    #: source of truth, never re-typed in a test or spec.
    keywordDescription: ClassVar[str]
