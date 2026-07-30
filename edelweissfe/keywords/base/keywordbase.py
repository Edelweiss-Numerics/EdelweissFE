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

from abc import ABC, abstractmethod
from typing import ClassVar

from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import OptionSchemaProvider


class KeywordBase(OptionSchemaProvider, ABC):
    """Base class for top-level ``.inp`` keywords (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U1).

    This generalizes the shape already proven by :class:`~edelweissfe.generators.base.
    generatorbase.GeneratorBase` (structural, no further interface beyond populating the model) and
    :class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase` (an object with further
    behaviour, heterogeneous per-module structural arguments) to cover the *keyword* level itself --
    the structural mesh/job keywords (``*element``, ``*node``, ``*nSet``, ``*elSet``, ``*surface``,
    ``*job``), the pluggable-module keywords (``*output``, ``*section``, ``*analyticalField``,
    ``*constraint``, ``*modelGenerator``, ``*step``), and the provider-dispatched type keywords
    (``*element``, ``*material``). It is not a new pattern, only the same one pushed one level up,
    discovered via the reserved ``"keyword"`` category of
    :mod:`edelweissfe.config.registry` rather than being hand-declared in
    ``inputfileparser.py``.

    A keyword is reached either from Python or from an ``.inp`` file, and the input file is a
    *serialization* of the Python path, not a second way of building whatever the keyword
    constructs. A ``KeywordBase`` subclass therefore carries an L2 :attr:`schema` -- the grammar's
    single source of truth, built from :func:`~edelweissfe.utils.schema.schemaField`,
    :func:`~edelweissfe.utils.schema.subKeywordField` and
    :func:`~edelweissfe.utils.schema.datalineField` -- plus one L4 seam,
    :meth:`fromKeywordDefinition`, the only place the ``.inp`` front-end's string-typed shape is
    turned into whatever this keyword produces.

    Unlike :class:`GeneratorBase`, ``KeywordBase`` declares no constructor shape at all: a
    structural keyword mutates ``context.model`` directly and returns ``None`` (nothing further is
    ever looked up by name), while a pluggable-module keyword resolves a further ``type``/
    ``provider`` value via the registry and constructs (or looks up) an object of some other,
    unrelated base class entirely (a ``Section``, a ``Solver``, an ``OutputManager``, ...). The one
    thing every keyword has in common is the seam below, not a shared runtime interface -- so this
    class declares no ``__init__``, abstract or otherwise.
    """

    #: The L2 schema dataclass describing this keyword's own line options, dataline payload, and
    #: ``>>`` sub-blocks, or ``None`` if it declares none (yet). See
    #: :class:`~edelweissfe.utils.schema.OptionSchemaProvider`.
    schema: ClassVar[type | None] = None

    @classmethod
    @abstractmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Create (or apply) this keyword from a parsed ``.inp`` keyword definition.

        This is the L4 seam: the one place a single occurrence of this keyword's input-file shape
        (line options + datalines + ``>>`` sub-blocks) is turned into typed arguments, validated
        and coerced against :attr:`schema` (typically via
        :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`, or
        ``schema.fromDatalines(...)`` for a :class:`~edelweissfe.utils.schema.
        DatalineAggregatingSchema`-shaped payload), resolves any further ``type``/``provider``
        dispatch through the registry, and constructs or applies the resulting object.

        A structural keyword (``*element``, ``*node``, ``*nSet``, ``*elSet``, ``*surface``,
        ``*job``) mutates ``context.model`` directly, like a
        :meth:`~edelweissfe.generators.base.generatorbase.GeneratorBase.fromGeneratorDefinition`
        call, and returns ``None`` -- there is nothing further for a caller to hold onto by name. A
        pluggable-module keyword (``*output``, ``*section``, ...) instead returns the object it
        constructed (or looked up), for the caller to register wherever that kind of object lives
        (an output-manager list, ``context.model``'s sections, ...).

        Parameters
        ----------
        name
            The name of this keyword occurrence (the keyword's own ``name=`` option, where one is
            declared; some structural keywords are unnamed and this may be a parser-assigned
            placeholder instead).
        definition
            The raw parsed block for *one* occurrence of this keyword: its line options, its
            datalines, and its ``>>`` sub-blocks, in whatever shape the ``.inp`` parser produces
            (mirroring the ``definition`` parameter of
            :meth:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase.
            fromStepActionDefinition` and
            :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.
            fromConstraintDefinition`).
        context
            The :class:`~edelweissfe.utils.inputcontext.InputContext` carrying the collaborators
            (model, journal, field-output controller, plotter) an L4 adapter needs.

        Returns
        -------
        KeywordBase | None
            The constructed (or looked-up) object, or ``None`` for a structural keyword that only
            mutates ``context.model`` in place.
        """
