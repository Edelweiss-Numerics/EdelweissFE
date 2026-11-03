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
Renders EdelweissFE's current input-language surface to stdout, sourced entirely from the registry
and the option schemas it exposes. This is the schema-sourced successor of the legacy renderer,
which walked ``edelweissfe.utils.inputlanguage.InputLanguage``'s ``Module`` tree
(``Module.__doc__``/``InputFileKeyword.__doc__``/``printKeywords()``) -- that mechanism, and this
script's own former dependency on it, has since been removed. The output format is unchanged (this
is what makes ``tests/golden/inputlanguage_surface.txt`` still the correct oracle): two sections,
``===== printKeywords() =====`` for the 23 top-level keywords, then one ``===== module
documentation: <dotted path> =====`` block per pluggable module, sorted by dotted module path.

Two rendering formats, both reproduced by :mod:`edelweissfe.utils.schemasurface`:

- :func:`~edelweissfe.utils.schemasurface.renderPrintKeywordsBlock` for the top ``printKeywords()``
  section, over the 23 ``KeywordBase`` subclasses registered under the registry's ``"keyword"``
  category (``name``/``description``/``schema`` all read off the class via
  :func:`~edelweissfe.utils.schemasurface.specFromKeywordClass` -- no hand-typed spelling).
- :func:`~edelweissfe.utils.schemasurface.renderSchemaSurface` for every module-doc section, one
  :class:`~edelweissfe.utils.schemasurface.KeywordSurfaceSpec` per module, prefixed by the module's
  own ``__doc__`` (unchanged from the legacy renderer's ``if mod.__doc__: ...`` behaviour).

A module's own ``name``/``description`` text is not recoverable from any live object now that the
``Module``/``InputFileKeyword`` declarations that used to carry it have been removed -- unlike a
top-level keyword, whose ``KeywordBase`` subclasses already carry
``keywordName``/``keywordDescription`` for it. Every other category (``constraint``, ``generator``,
``outputmanager``, ``analyticalfield``, ``section``, ``stepaction``, ``modelmodifier``) therefore
has its ``name``/``description`` pair transcribed once, literally, into the spec tables below --
the same "transcribed verbatim from the golden" convention already used for step
actions in ``tests/test_schemasurface.py``'s ``_U3B_STEP_ACTION_SPECS`` (this module's step-action
table is transcribed from that same source). The registry still supplies the one thing that must
never drift by hand-transcription: the *schema* itself, resolved fresh via ``registry.lookup``, so
the option/dataline/sub-keyword content is always the live, current grammar -- only the cosmetic
``name``/``description`` header text is a literal.

A few sections have no schema to render at all (``schema=None``, by design -- see each module's own
docstring) but still carry real golden content: ``generators.executepythoncode``'s raw-code
dataline and ``stepactions.options``' dispatcher keyword. Both get a small, explicitly-documented
"minimal render path" below (a tiny local schema built only for rendering, never imported by any
runtime code) rather than being silently skipped. ``analyticalfields.mapped`` and
``outputmanagers.meshplot`` were never ``Module``-documented in the first place -- both carry a
hand-maintained ``documentation = {...}`` dict as their sole source of truth (deliberately kept,
per each module's own docstring) -- rendered via
:func:`~edelweissfe.utils.schemasurface.renderDictDocumentation`.

Runs as a **fresh interpreter process** (see ``test_inputlanguage_golden.py``), matching a real
simulation run or the Sphinx doc build; nothing here depends on import order any more (the schema
registry, unlike the old ``Module`` tree, has no ``if keyword in inputLanguage:`` silent-no-op
guard), but keeping this a subprocess still isolates the rendered surface from whatever other test
modules happen to run in the same pytest session.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from edelweissfe.config import registry
from edelweissfe.utils.schema import datalineField, schemaField, schemaFields
from edelweissfe.utils.schemasurface import (
    KeywordSurfaceSpec,
    renderDictDocumentation,
    renderPrintKeywordsBlock,
    renderSchemaSurface,
    specFromKeywordClass,
)

# ===== printKeywords() section: the 23 top-level keywords ========================================

#: Exact rendering order of ``inputfileparser.py``'s (now-deleted) ``inputLanguage.addKeyword(...)``
#: call sequence -- NOT alphabetical, and NOT the order ``config/registry.py``'s ``"keyword"``
#: category dict happens to be declared in (that dict groups ``job`` next to the other structural
#: keywords; the legacy parser declared it later, interleaved with ``section``/``material``). This
#: list is the single place that order is pinned now that no ``Module`` tree exists to iterate.
#: ``elementProperty`` and ``restart`` have no legacy position -- both were added after the U2 gate
#: and are appended last, in the order they were actually added.
_TOP_LEVEL_KEYWORDS_IN_LEGACY_ORDER = [
    "element",
    "elSet",
    "node",
    "nSet",
    "surface",
    "section",
    "material",
    "advancedmaterial",
    "fieldOutput",
    "analyticalField",
    "job",
    "solver",
    "step",
    "output",
    "updateConfiguration",
    "modelGenerator",
    "constraint",
    "modelModifier",
    "configurePlots",
    "exportPlots",
    "include",
    "elementProperty",
    "restart",
]


def _renderPrintKeywordsSection() -> str:
    """Render the ``===== printKeywords() =====`` section from the registry's ``"keyword"``
    category, in :data:`_TOP_LEVEL_KEYWORDS_IN_LEGACY_ORDER`.

    Blocks are joined by ``"\\n\\n\\n"`` -- two blank lines -- matching ``printKeywords()``'s own
    ``print(wrapper.fill(...))`` (one trailing newline) followed by ``print("\\n")`` (the string
    ``"\\n"`` plus ``print``'s own terminator, i.e. two more newlines) between keywords.
    """
    blocks = []
    for keywordName in _TOP_LEVEL_KEYWORDS_IN_LEGACY_ORDER:
        target, _schema = registry.lookup("keyword", keywordName)
        blocks.append(renderPrintKeywordsBlock(specFromKeywordClass(target)))
    return "\n\n\n".join(blocks)


# ===== module documentation sections ==============================================================


@dataclass(frozen=True)
class _BracketModuleSpec:
    """One ``[name] description`` ("bracket"-headed) module documentation section, transcribed from
    the golden file -- every category except step actions and the ``>>``-only ``utils.fieldoutput``
    (see :data:`_STEP_ACTION_SPECS`/:func:`_fieldOutputSection`, both of which use the angle-bracket
    ``< name >`` header instead) fits this shape: one keyword, resolved once from its own registry
    entry, rendered without repetition.
    """

    modpath: str
    category: str
    registryName: str
    displayName: str
    description: str

    def schema(self) -> type | None:
        _target, schema = registry.lookup(self.category, self.registryName)
        return schema

    def render(self) -> str:
        return renderSchemaSurface(
            [KeywordSurfaceSpec(name=self.displayName, description=self.description, schema=self.schema())]
        )


#: Transcribed verbatim from ``tests/golden/inputlanguage_surface.txt``'s ``[name] description``
#: header lines -- the ``name``/``description`` pair a (now-deleted) ``Module(name, description)``/
#: ``addOptionalKeyword(name, description)`` call used to own. The schema for each is resolved fresh
#: from the registry (never transcribed), so only the cosmetic header text is a literal here.
_BRACKET_MODULE_SPECS = [
    _BracketModuleSpec(
        "edelweissfe.analyticalfields.fromvtk",
        "analyticalfield",
        "fromvtk",
        "fromVtk",
        "Use PyVista to interpolate from vtk data.",
    ),
    _BracketModuleSpec(
        "edelweissfe.analyticalfields.randomscalar",
        "analyticalfield",
        "randomscalar",
        "randomScalar",
        "Define a random field using the GSTools library.",
    ),
    _BracketModuleSpec(
        "edelweissfe.analyticalfields.scalarexpression",
        "analyticalfield",
        "scalarexpression",
        "scalarExpression",
        "Define an analytical field using a scalar expression.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.amrtransparencyprobe",
        "constraint",
        "amrtransparencyprobe",
        "amrtransparencyprobe",
        "Test-only acceptance double verifying that a cached NodeSet reference survives AMR without "
        "any observer registration.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.directionalspringpenalty",
        "constraint",
        "directionalspringpenalty",
        "directionalSpringPenalty",
        "A penalty based constraint used for assigning a specific stiffness to the nodes of a " "defined node set.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.equalvaluelagrangian",
        "constraint",
        "equalvaluelagrangian",
        "equalvaluelagrangian",
        "A lagrangian multiplier based constraint used for constraining nodal values of a node set " "to be equal.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.equalvaluepenalty",
        "constraint",
        "equalvaluepenalty",
        "equalvaluepenalty",
        "A lagrangian multiplier based constraint used for constraining nodal values of a node set " "to be equal.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.hangingnode",
        "constraint",
        "hangingnode",
        "hangingnode",
        "Hanging-node MPC (DOF elimination) tying refined-side nodes to the coarse serendipity trace.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.linearizedrigidbody",
        "constraint",
        "linearizedrigidbody",
        "linearizedrigidbody",
        "A rigid body constraint tying nodes to a reference point.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.nodetodeformablesurfacepenalty",
        "constraint",
        "nodetodeformablesurfacepenalty",
        "nodeToDeformableSurfacePenalty",
        "A penalty based unilateral contact constraint preventing the nodes of a slave surface from "
        "penetrating a deformable master surface, both represented by contact facet elements.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.nodetodiscreterigidbodypenalty",
        "constraint",
        "nodetodiscreterigidbodypenalty",
        "nodeToDiscreteRigidBodyPenalty",
        "A penalty based unilateral contact constraint preventing nodes of a node set from "
        "penetrating the surface of a discrete rigid body.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.nodetorigidsurfacepenalty",
        "constraint",
        "nodetorigidsurfacepenalty",
        "nodeToRigidSurfacePenalty",
        "A penalty based unilateral constraint used for preventing the nodes of a node set from "
        "penetrating a defined rigid boundary.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.penaltyindirectcontrol",
        "constraint",
        "penaltyindirectcontrol",
        "penaltyindirectcontrol",
        "A penalty based constraint used for indirect (displacement) control.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.rigidbody",
        "constraint",
        "rigidbody",
        "rigidbody",
        "A rigid body constraint tying nodes to a reference point.",
    ),
    _BracketModuleSpec(
        "edelweissfe.constraints.tie",
        "constraint",
        "tie",
        "tie",
        "An Abaqus-style tie constraint, bonding a slave surface rigidly to a deformable master "
        "surface via master-slave DOF elimination.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.boxgen",
        "generator",
        "boxgen",
        "boxgen",
        "A mesh generator for cuboid geometries and structured hex meshes.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.cubit",
        "generator",
        "cubit",
        "cubit",
        "Interface to Cubit. Generate mesh using Cubit .jou files.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.cuboidlatticegenerator",
        "generator",
        "cuboidlatticegenerator",
        "cuboidlatticegenerator",
        "A mesh generator for generating cuboid lattice structure.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.discreterigidbodygenerator",
        "generator",
        "discreterigidbodygenerator",
        "discreteRigidBodyGenerator",
        "Generates a discrete rigid body from a surface mesh file (Exodus, STL, OBJ, or any other "
        "format readable by PyVista).",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.findclosestnode",
        "generator",
        "findclosestnode",
        "findclosestnode",
        "Find the node closest to a given spatial position, and store it in an existing or new node " "set.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.microstructuregenerator",
        "generator",
        "microstructuregenerator",
        "microstructuregenerator",
        "A mesh generator for generating a structure from a single unit cell mesh.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.pipegen",
        "generator",
        "pipegen",
        "pipegen",
        "A structured hex mesh generator for pipe geometries.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.planerectquad",
        "generator",
        "planerectquad",
        "planeRectQuad",
        "A mesh generator for cuboid geometries and structured hex meshes.",
    ),
    _BracketModuleSpec(
        "edelweissfe.generators.surfaceelementgenerator",
        "generator",
        "surfaceelementgenerator",
        "surfaceElementGenerator",
        "Generates flat contact facet elements (Tria3ContactFacet/Line2ContactFacet) from an "
        "existing *surface definition.",
    ),
    _BracketModuleSpec(
        "edelweissfe.modelmodifiers.adaptivity.hadaptivity",
        "modelmodifier",
        "hadaptivity",
        "hadaptivity",
        "Dynamic hanging-node h-adaptivity model modifier for HEX20 elements.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.computetimemonitor",
        "outputmanager",
        "computetimemonitor",
        "computetimemonitor",
        "A simple monitor to observe results (fieldOutputs) in the console during analysis.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.conditionalstop",
        "outputmanager",
        "conditionalstop",
        "ConditionalStop",
        "A simple monitor to observe results (fieldOutputs) in the console during analysis.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.ensight",
        "outputmanager",
        "ensight",
        "ensight",
        "Ensight export.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.fractureenergyintegrator",
        "outputmanager",
        "fractureenergyintegrator",
        "fractureEnergyIntegrator",
        "A simple integrator to compute the fracture energy by integrating a load-displacement curve.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.meshdatatofile",
        "outputmanager",
        "meshdatatofile",
        "meshdatatofile",
        "Writes the (generated) mesh data to a file.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.monitor",
        "outputmanager",
        "monitor",
        "monitor",
        "A simple monitor to observe results (fieldOutputs) in the console during analysis.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.plotalongpath",
        "outputmanager",
        "plotalongpath",
        "plotAlongPath",
        "Plot result for a nodeSet or an elementSet along the true geometrical distance.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.restart",
        "outputmanager",
        "restart",
        "restart",
        "Writes restart checkpoints during the analysis.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.statusfile",
        "outputmanager",
        "statusfile",
        "statusfile",
        "Writes a status file during the analysis.",
    ),
    _BracketModuleSpec(
        "edelweissfe.outputmanagers.timemonitor",
        "outputmanager",
        "timemonitor",
        "timemonitor",
        "Writes the model time at the end of each increment to a file.",
    ),
    _BracketModuleSpec(
        "edelweissfe.sections.plane",
        "section",
        "plane",
        "plane",
        "This section represents a classical plane solid materal section.",
    ),
    _BracketModuleSpec(
        "edelweissfe.sections.solid",
        "section",
        "solid",
        "solid",
        "This section represents a classical solid materal section.",
    ),
]


# --- step actions: the ``< name >``/``< updateName >`` shape, repeated once per registered step ---
# type (the legacy ``for module in modules: kw = module.addOptionalKeyword(...)`` loop iterated over
# every step type's own ``Module``, so the same pair rendered once per type). Transcribed from the
# same golden source as ``tests/test_schemasurface.py``'s ``_U3B_STEP_ACTION_SPECS``.


@dataclass(frozen=True)
class _StepActionModuleSpec:
    """One step action's expected ``< name >``/``< updateName >`` module documentation section."""

    modpath: str
    registryName: str
    name: str
    description: str
    updateName: str | None = None
    updateDescription: str | None = None
    updateSchemaAttr: str | None = None

    def render(self) -> str:
        _target, schema = registry.lookup("stepaction", self.registryName)
        updateSchema = None
        if self.updateSchemaAttr is not None:
            mod = importlib.import_module(self.modpath)
            updateSchema = getattr(mod, self.updateSchemaAttr)

        specs = []
        for _ in range(_stepTypeCount()):
            specs.append(KeywordSurfaceSpec(name=self.name, description=self.description, schema=schema))
            if updateSchema is not None:
                specs.append(
                    KeywordSurfaceSpec(name=self.updateName, description=self.updateDescription, schema=updateSchema)
                )
        return renderSchemaSurface(specs, bracket="<", joiner="\n")


def _stepTypeCount() -> int:
    """How many times a step action's ``< name >``/``< updateName >`` block pair repeats: once per
    registered ``"step"`` type, exactly mirroring the legacy ``for module in modules:`` loop that
    used to register the same step action onto every step type's own ``Module``."""
    return len(registry.availableNames("step"))


_STEP_ACTION_SPECS = [
    _StepActionModuleSpec(
        "edelweissfe.stepactions.bodyforce", "bodyforce", "bodyforce", "Apply body forces on element sets."
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.changematerialproperty",
        "changematerialproperty",
        "changematerialproperty",
        "Stepaction to change material properties.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.dirichlet",
        "dirichlet",
        "dirichlet",
        "Standard Dirichlet boundary condition.",
        "updateDirichlet",
        "Update a previously defined dirichlet definition.",
        "UpdateDirichletSchema",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.distributedload",
        "distributedload",
        "distributedload",
        "Standard distributed load, applied on a surface set.",
        "updatedistributedload",
        "Update a previously defined distributedload definition.",
        "UpdateDistributedloadSchema",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.geostatic",
        "geostatic",
        "geostatic",
        "Initialize materials to an geostatic stress state.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.indirectcontractioncontrol",
        "indirectcontractioncontrol",
        "indirectcontractioncontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control "
        "the contraction, e.g., for tunneling simulations.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.indirectcontrol",
        "indirectcontrol",
        "indirectcontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control "
        "the contraction, e.g., for tunneling simulations.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.initializematerial",
        "initializematerial",
        "initializematerial",
        "Standard distributed load, applied on a surface set.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.modelupdate",
        "modelupdate",
        "modelupdate",
        "This step action may be used for updating the model at the beginning of a step.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.nodeforces",
        "nodeforces",
        "nodeforces",
        "Apply node forces on node sets.",
        "updateNodeforces",
        "Update a previously defined nodeforces definition.",
        "UpdateNodeforcesSchema",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.setfield",
        "setfield",
        "setfield",
        "Set a field (via fieldOutput) to a predefined value.",
    ),
    _StepActionModuleSpec(
        "edelweissfe.stepactions.setinitialconditions",
        "setinitialconditions",
        "setinitialconditions",
        "Pass initial conditions to elements.",
    ),
]


# --- the schema=None-but-content-bearing sections: a small, explicitly documented "minimal render --
# path" for each. Neither schema below is imported by any runtime code -- both exist solely so this
# renderer can reproduce the corresponding golden section without a real (non-``None``) schema to
# resolve from the registry.


@dataclass(frozen=True)
class _ExecutePythonCodeDocSchema:
    """Rendering-only stand-in for ``generators.executepythoncode``'s grammar: its datalines are raw
    Python source (see that module's ``Generator.schema = None`` and its own docstring), so there is
    no flat option schema to resolve from the registry -- only this dataline description, transcribed
    from the golden file."""

    code: str = datalineField(description="Python code to run", required=True)


def _executePythonCodeSection() -> str:
    from edelweissfe.generators.executepythoncode import Generator

    return renderSchemaSurface(
        [
            KeywordSurfaceSpec(
                name="executePythoncode", description=Generator.__doc__, schema=_ExecutePythonCodeDocSchema
            )
        ]
    )


def _stepActionOptionsSection() -> str:
    """Rendering-only stand-in for ``stepactions.options``' ``< options >`` keyword: it is a
    dispatcher onto another object's own schema, resolved dynamically at runtime by ``name`` (see
    that module's own docstring), so it declares no schema of its own -- only the one static
    argument (``name``) the parser enforces up front is rendered here, transcribed from the golden
    file, repeated once per registered step type exactly like every other step action."""
    from edelweissfe.stepactions.options import _OPTIONS_KEYWORD_DESCRIPTION

    @dataclass(frozen=True)
    class _OptionsDocSchema:
        name: str = schemaField(
            description="The name of the already-declared solver or output manager this block configures.",
            dtype=str,
            default=None,
            required=True,
        )

    specs = [
        KeywordSurfaceSpec(name="options", description=_OPTIONS_KEYWORD_DESCRIPTION, schema=_OptionsDocSchema)
        for _ in range(_stepTypeCount())
    ]
    return renderSchemaSurface(specs, bracket="<", joiner="\n")


# --- utils.fieldoutput: the ``*fieldOutput`` keyword's own hosted grammar. Unlike every other -------
# pluggable-module keyword, it carries no top-level ``[fieldOutput] ...`` header line of its own at
# all in the golden -- its three ``>>`` blocks are documented directly, one ``< name >`` header each,
# joined with a single newline (no blank-line separator) -- because the legacy ``documentation`` list
# here held three bare ``InputFileKeyword``s, never wrapped in a ``Module``.


def _fieldOutputSection() -> str:
    from edelweissfe.keywords.fieldoutput import FieldOutputKeyword

    meta = schemaFields(FieldOutputKeyword.schema)
    specs = [
        KeywordSurfaceSpec(name=fieldName, description=fieldMeta.description, schema=fieldMeta.subSchema)
        for fieldName, fieldMeta in meta.items()
    ]
    return renderSchemaSurface(specs, bracket="<", joiner="\n")


# --- the two dict-style ("legacy dict documentation") modules: never Module-documented in the -------
# first place, so there is no schema-vs-dict migration to do here at all -- both keep their own
# hand-maintained ``documentation = {...}`` dict as sole source of truth (see each module's own
# docstring), rendered via ``renderDictDocumentation``.


def _mappedSection() -> str:
    import edelweissfe.analyticalfields.mapped as mapped

    return renderDictDocumentation(mapped.documentation)


def _meshplotSection() -> str:
    import edelweissfe.outputmanagers.meshplot as meshplot

    return renderDictDocumentation(meshplot.documentation)


# ===== assembling the whole surface, sorted by dotted module path, exactly as the legacy ==========
# ``sorted(discovered, key=lambda pair: pair[0])`` did.


def _moduleDocSections() -> list[tuple[str, str]]:
    """Every ``(dotted module path, rendered grammar text)`` pair, unsorted -- the grammar text is
    everything AFTER the module's own ``__doc__`` line(s), matching the legacy renderer's own
    ``_renderDocumentation`` step."""
    sections: dict[str, str] = {}
    for spec in _BRACKET_MODULE_SPECS:
        sections[spec.modpath] = spec.render()
    for spec in _STEP_ACTION_SPECS:
        sections[spec.modpath] = spec.render()
    sections["edelweissfe.stepactions.options"] = _stepActionOptionsSection()
    sections["edelweissfe.generators.executepythoncode"] = _executePythonCodeSection()
    sections["edelweissfe.utils.fieldoutput"] = _fieldOutputSection()
    sections["edelweissfe.analyticalfields.mapped"] = _mappedSection()
    sections["edelweissfe.outputmanagers.meshplot"] = _meshplotSection()
    return sorted(sections.items(), key=lambda pair: pair[0])


def renderCurrentInputLanguageSurface() -> str:
    """Render the whole surface: the ``printKeywords()`` section, then one
    ``===== module documentation: <path> =====`` block per pluggable module, sorted by dotted path.
    """
    parts = ["===== printKeywords() =====", _renderPrintKeywordsSection()]

    for modpath, grammar in _moduleDocSections():
        parts.append(f"===== module documentation: {modpath} =====")
        mod = importlib.import_module(modpath)
        if mod.__doc__:
            parts.append(mod.__doc__.strip())
        parts.append(grammar)

    # Strip per-line trailing whitespace -- printKeywords()'s textwrap-based rendering pads columns
    # via trailing spaces for a description-less argument. Without this normalisation the
    # `trailing-whitespace` pre-commit hook would rewrite the committed golden file on every commit,
    # permanently breaking this comparison.
    rendered = "\n".join(parts)
    return "\n".join(line.rstrip() for line in rendered.split("\n")) + "\n"


if __name__ == "__main__":
    import sys

    if "--list-modules" in sys.argv:
        # The dotted module paths only, as JSON, for tests/test_module_import_independence.py. It
        # shares this list rather than repeating it, so the two cannot disagree about what "a
        # documented module" is -- if the golden surface covers a module, the import gate must cover
        # it too.
        import json

        print(json.dumps([modpath for modpath, _ in _moduleDocSections()]))
    else:
        print(renderCurrentInputLanguageSurface(), end="")
