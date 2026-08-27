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

"""Lazy registry mapping ``(category, name)`` to the object that implements it.

Used both by the ``.inp`` parser (to resolve a keyword/type name to its implementing class) and
directly by third-party packages (EdelweissMeshfree, plugins) that need the same objects without
going through the parser.

Four properties are load-bearing; the first three are each covered by a dedicated test in
``tests/test_registry.py``:

1. **Zero eager imports.** Importing this module must not import any element, material, solver,
   or output-manager module -- it only ever imports :mod:`importlib` and
   :mod:`importlib.metadata`. Resolving a name (importing the module that actually implements it)
   happens exclusively inside :func:`lookup`, on first use of that particular ``(category,
   name)`` pair.
2. **Built-ins work with a stale editable install.** Entry points declared in a package's
   ``pyproject.toml`` only materialize in ``importlib.metadata`` after ``pip install -e .`` is
   re-run, so EdelweissFE's own modules are additionally listed in the static ``_BUILTINS`` table
   below as plain ``"module.path:AttrName"`` strings, requiring no metadata regeneration. Entry
   points are layered *on top* of that table for third-party discovery.
3. **Thread-safe memoization.** ``lookup()`` caches the resolved object in ``_resolved`` so a
   given ``(category, name)`` is only imported once. Under free-threading (``PYTHON_GIL=0``),
   multiple threads can call ``lookup()`` for the same key concurrently; see the docstring of
   :func:`lookup` for the chosen strategy.
4. **Names are case-insensitive.** Both ``category`` and ``name`` are casefolded on the way in
   (:func:`lookup`, :func:`register`) and stored casefolded in ``_BUILTINS``, so a name resolves
   the same way whether it arrives through the ``.inp`` parser or directly from an external
   package. Casefolded keys widen the chance of two callers claiming the same name, which is why
   every way of claiming a name raises :class:`RegistryConflictError` instead of silently
   overwriting -- see that class.

Built-in coverage
------------------
The ``_BUILTINS`` table below covers these categories:

``outputmanager`` (10), ``section`` (3), ``constraint`` (12), ``stepaction`` (13),
``generator`` (10), ``analyticalfield`` (3), ``solver`` (7), ``step`` (2), ``modelmodifier`` (1),
``statetransferstrategy`` (3), ``element`` (42), ``material`` (7), ``linsolver`` (11).

``keyword`` is the single source the ``.inp`` parser consults for every top-level keyword
(``element``, ``node``, ``nSet``, ``elSet``, ``surface``, ``job``, ``section``, ``elementProperty``,
``material``, ``advancedmaterial``, ``fieldOutput``, ``analyticalField``, ``solver``, ``step``,
``output``, ``updateConfiguration``, ``modelGenerator``, ``constraint``, ``modelModifier``,
``configurePlots``, ``exportPlots``, ``include``), each mapped to a
:class:`edelweissfe.keywords.base.keywordbase.KeywordBase` subclass.

``element`` and ``material`` are keyed by *element type* / *material name* and cover exactly the
``provider=edelweiss`` namespace of ``config/elementlibrary.py`` and ``config/materiallibrary.py``.
The ``provider`` axis itself is not part of this registry: it selects a namespace rather than a
variant of one lookup, and only ``edelweiss`` addresses anything by name -- ``marmot`` and
``marmotsingleqpelement`` ignore the name and return a single wrapper class, ``marmotmaterial``
returns ``None``. Those branches stay an explicit table in their own module.

``linsolver`` is covered by a uniform ``createSolver(opts) -> Callable[[A, b], x]`` factory per
solver (see the ``linsolver`` block below), since not every linear solver reduces to a plain
class/callable a dotted string can point at.

What is outside the registry:

- The ``provider`` axis of ``config/elementlibrary.py``/``config/materiallibrary.py`` (see above).
- ``config/solvers.py``'s ``solverLibrary``, ``config/outputmanagers.py``, ``config/sections.py``
  and ``config/analyticalfields.py``, which exist purely as Sphinx rendering targets and resolve
  nothing themselves.
"""

from __future__ import annotations

import importlib
import threading
from importlib.metadata import entry_points
from typing import Any

from edelweissfe.utils.misc import findSimilarString
from edelweissfe.utils.schema import schemaOf

#: Entry-point group name third-party packages register EdelweissFE-discoverable implementations
#: under, e.g. in a plugin's ``pyproject.toml``::
#:
#:     [project.entry-points."edelweissfe.plugins"]
#:     "outputmanager.myoutputmanager" = "mypackage.mymodule:MyOutputManager"
ENTRY_POINT_GROUP = "edelweissfe.plugins"


class RegistryLookupError(LookupError):
    """Raised by :func:`lookup` when no implementation is registered for a requested name.

    Unlike a bare ``findSimilarString`` call (which raises a plain ``Exception`` with the message
    "You tried to find a string similar to ... in an empty list." when the candidate list happens
    to be empty), this always produces a message naming the category and, whenever at least one
    name is registered for it, a "did you mean" suggestion.
    """


class RegistryConflictError(RegistryLookupError):
    """Raised when two different implementations claim the same ``(category, name)``.

    A :class:`RegistryLookupError` subclass, so a caller guarding a name resolution with
    ``except RegistryLookupError`` also catches a conflict rather than letting it escape as an
    unrelated exception type. Re-registering the identical object under the same name is not a
    conflict (identity is compared before raising) and is idempotent.
    """


#: Static built-in table: ``(category, name)`` (both stored casefolded) -> ``"module.path:Attr"``.
#: Never imports anything by itself -- it is a table of strings, resolved lazily by
#: :func:`_resolveDottedString` only when :func:`lookup` actually needs that entry.
_BUILTINS: dict[tuple[str, str], str] = {}


def _addBuiltins(category: str, attrName: str, moduleNames: list[str], packageDotted: str) -> None:
    """Populate ``_BUILTINS`` for a category following the uniform "one module per name, fixed
    attribute name" convention used by most ``edelweissfe`` subpackages.

    Parameters
    ----------
    category
        The registry category these entries belong to (e.g. ``"outputmanager"``).
    attrName
        The attribute name to look up in each resolved module (e.g. ``"OutputManager"``).
    moduleNames
        The submodule names (without package prefix), one per registrable name.
    packageDotted
        The dotted package path the submodules live in (e.g. ``"edelweissfe.outputmanagers"``).
    """
    for moduleName in moduleNames:
        _BUILTINS[(category, moduleName.casefold())] = f"{packageDotted}.{moduleName}:{attrName}"


_addBuiltins(
    "outputmanager",
    "OutputManager",
    [
        "computetimemonitor",
        "conditionalstop",
        "ensight",
        "fractureenergyintegrator",
        "meshdatatofile",
        "meshplot",
        "monitor",
        "plotalongpath",
        "statusfile",
        "timemonitor",
    ],
    "edelweissfe.outputmanagers",
)

_addBuiltins(
    "section",
    "Section",
    ["plane", "planerandomthickness", "solid"],
    "edelweissfe.sections",
)

_addBuiltins(
    "constraint",
    "Constraint",
    [
        "amrtransparencyprobe",
        "directionalspringpenalty",
        "equalvaluelagrangian",
        "equalvaluepenalty",
        "hangingnode",
        "linearizedrigidbody",
        "nodetodeformablesurfacepenalty",
        "nodetodiscreterigidbodypenalty",
        "nodetorigidsurfacepenalty",
        "penaltyindirectcontrol",
        "rigidbody",
        "tie",
    ],
    "edelweissfe.constraints",
)

_addBuiltins(
    "stepaction",
    "StepAction",
    [
        "bodyforce",
        "changematerialproperty",
        "dirichlet",
        "distributedload",
        "geostatic",
        "indirectcontractioncontrol",
        "indirectcontrol",
        "initializematerial",
        "modelupdate",
        "nodeforces",
        "options",
        "setfield",
        "setinitialconditions",
    ],
    "edelweissfe.stepactions",
)

_addBuiltins(
    "generator",
    "Generator",
    [
        "boxgen",
        "cubit",
        "cuboidlatticegenerator",
        "discreterigidbodygenerator",
        "executepythoncode",
        "findclosestnode",
        "microstructuregenerator",
        "pipegen",
        "planerectquad",
        "surfaceelementgenerator",
    ],
    "edelweissfe.generators",
)

_addBuiltins(
    "analyticalfield",
    "AnalyticalField",
    ["fromvtk", "randomscalar", "scalarexpression"],
    "edelweissfe.analyticalfields",
)

# solver / step / modelmodifier / statetransferstrategy are not "one module per name" -- this table
# is their only copy. config/solvers.py's Sphinx ``.. pprint::`` directive reads this registry
# directly rather than keeping its own copy.
for _solverName, _moduleName in {
    "NIST": "nonlinearimplicitstatic",
    "NEST": "nonlinearexplicitstatic",
    "NED": "nonlinearexplicitdynamic",
    "NISTParallel": "nonlinearimplicitstaticparallel",
    "NESTParallel": "nonlinearexplicitstaticparallel",
    "NEDParallel": "nonlinearexplicitdynamicparallel",
    "NISTPArcLength": "nonlinearimplicitstaticparallelarclength",
}.items():
    _BUILTINS[("solver", _solverName.casefold())] = f"edelweissfe.solvers.{_moduleName}:{_solverName}"

_BUILTINS[("step", "adaptive")] = "edelweissfe.steps.adaptivestep:AdaptiveStep"
_BUILTINS[("step", "adaptiveforexplicitsimulations")] = (
    "edelweissfe.steps.adaptivestepforexplicitsimulations:AdaptiveStepForExplicitSimulations"
)

_BUILTINS[("modelmodifier", "hadaptivity")] = "edelweissfe.modelmodifiers.adaptivity.hadaptivity:ModelModifier"

_BUILTINS[("statetransferstrategy", "nearestqp")] = "edelweissfe.adaptivity.statetransfer:NearestQuadraturePointCopy"
_BUILTINS[("statetransferstrategy", "projection")] = "edelweissfe.adaptivity.statetransfer:PolynomialProjection"
_BUILTINS[("statetransferstrategy", "virgin")] = "edelweissfe.adaptivity.statetransfer:VirginState"

# AMR refinement markers -- resolved by a ``>>marker, type=<name>`` block, one module (marking.py)
# per marker class. Kept a registry category, like statetransferstrategy, so any adaptivity
# mechanism (or a third-party package via entry points) constructs markers by name rather than
# through a hardcoded if/elif living inside one model modifier.
for _markerName, _markerClass in {
    "fieldOutput": "FieldOutputMarker",
    "elementSet": "ElementSetMarker",
    "nodeSet": "NodeSetMarker",
    "surface": "SurfaceMarker",
    "recoveryError": "RecoveryErrorMarker",
}.items():
    _BUILTINS[("marker", _markerName.casefold())] = f"edelweissfe.adaptivity.marking:{_markerClass}"

# The element category is keyed by element *type*, and 42 types share just 2 formulation classes, so
# `_addBuiltins`'s "one module per name, fixed attribute name" convention does not apply -- hence two
# explicit lists. This table is the single source of truth for type -> class, pinned against
# `elements/library.py`'s `elLibrary` by
# `tests/test_registry.py::test_element_category_covers_every_element_type_exactly`; the types are
# written out as literals rather than imported from `elements.library` so that this module keeps
# importing nothing (property 1 above).
for _elementTypeName in [
    "CPE4",
    "CPE4R",
    "CPE4E",
    "CPE4N",
    "CPE8",
    "CPE8R",
    "CPE8N",
    "CPS4",
    "CPS4R",
    "CPS4E",
    "CPS4N",
    "CPS8",
    "CPS8R",
    "CPS8N",
    "C3D8",
    "C3D8R",
    "C3D8E",
    "C3D8N",
    "C3D20",
    "C3D20R",
    "C3D20N",
]:
    _BUILTINS[("element", _elementTypeName.casefold())] = (
        "edelweissfe.elements.displacementelement.element:DisplacementElement"
    )

for _elementTypeName in [
    "CPE4TL",
    "CPE4RTL",
    "CPE4ETL",
    "CPE4NTL",
    "CPE8TL",
    "CPE8RTL",
    "CPE8NTL",
    "CPS4TL",
    "CPS4RTL",
    "CPS4ETL",
    "CPS4NTL",
    "CPS8TL",
    "CPS8RTL",
    "CPS8NTL",
    "C3D8TL",
    "C3D8RTL",
    "C3D8ETL",
    "C3D8NTL",
    "C3D20TL",
    "C3D20RTL",
    "C3D20NTL",
]:
    _BUILTINS[("element", _elementTypeName.casefold())] = (
        "edelweissfe.elements.displacementtlelement.element:DisplacementTLElement"
    )

# The material category covers the `provider=edelweiss` materials, one module and one class name per
# material -- so neither `_addBuiltins` variant fits and each entry names its own target. Originally
# transcribed from `config/materiallibrary.py`'s if/elif chain, which is now gone, so these are the
# only copy.
for _materialName, _materialDotted in {
    "linearelastic": "edelweissfe.materials.linearelastic.linearelastic:LinearElasticMaterial",
    "vonmises": "edelweissfe.materials.vonmises.vonmises:VonMisesMaterial",
    "neohooke": "edelweissfe.materials.neohooke.neohooke:NeoHookeanMaterial",
    "neohookeplastic": "edelweissfe.materials.neohookeplastic.neohookeplastic:NeoHookeanPlasticMaterial",
    "hyperelasticadvanced": (
        "edelweissfe.materials.hyperelasticadvanced.hyperelasticadvanced:HyperelasticAdvancedMaterial"
    ),
    "hyperelasticadvancedi2extended": (
        "edelweissfe.materials.hyperelasticadvanced.hyperelasticadvancedi2extended"
        ":HyperelasticAdvancedI2ExtendedMaterial"
    ),
    "hyperplasticadvanced": (
        "edelweissfe.materials.hyperplasticadvanced.hyperplasticadvanced:HyperplasticAdvancedMaterial"
    ),
}.items():
    _BUILTINS[("material", _materialName)] = _materialDotted

# The linsolver category: every `linsolve/*` subpackage exposes a module-level
# `createSolver(opts) -> Callable[[A, b], x]` factory. See each solver's own module docstring
# for what it does and how the optional-backend import is handled, and
# doc/source/documentation/linsolvers.rst for how the solvers relate to one another.
for _linsolverName in [
    "superlu",
    "umfpack",
    "pardiso",
    "panuapardiso",
    "klu",
    "petsclu",
    "mumps",
    "gmres",
    "amgcl",
    "blockamg",
    "matrixdump",
]:
    _BUILTINS[("linsolver", _linsolverName)] = f"edelweissfe.linsolve.{_linsolverName}:createSolver"


# The "keyword" category: each top-level keyword is its own module with its own class name, so,
# like the `material` table above, this is an explicit dict rather than `_addBuiltins`'s "one fixed
# attribute name" convention.
for _keywordName, _keywordDotted in {
    "element": "edelweissfe.keywords.element:ElementKeyword",
    "elSet": "edelweissfe.keywords.elset:ElSetKeyword",
    "node": "edelweissfe.keywords.node:NodeKeyword",
    "nSet": "edelweissfe.keywords.nset:NSetKeyword",
    "surface": "edelweissfe.keywords.surface:SurfaceKeyword",
    "job": "edelweissfe.keywords.job:JobKeyword",
    "section": "edelweissfe.keywords.section:SectionKeyword",
    "elementProperty": "edelweissfe.keywords.elementproperty:ElementPropertyKeyword",
    "material": "edelweissfe.keywords.material:MaterialKeyword",
    "advancedmaterial": "edelweissfe.keywords.advancedmaterial:AdvancedMaterialKeyword",
    "fieldOutput": "edelweissfe.keywords.fieldoutput:FieldOutputKeyword",
    "analyticalField": "edelweissfe.keywords.analyticalfield:AnalyticalFieldKeyword",
    "solver": "edelweissfe.keywords.solver:SolverKeyword",
    "step": "edelweissfe.keywords.step:StepKeyword",
    "output": "edelweissfe.keywords.output:OutputKeyword",
    "updateConfiguration": "edelweissfe.keywords.updateconfiguration:UpdateConfigurationKeyword",
    "modelGenerator": "edelweissfe.keywords.modelgenerator:ModelGeneratorKeyword",
    "constraint": "edelweissfe.keywords.constraint:ConstraintKeyword",
    "modelModifier": "edelweissfe.keywords.modelmodifier:ModelModifierKeyword",
    "configurePlots": "edelweissfe.keywords.configureplots:ConfigurePlotsKeyword",
    "exportPlots": "edelweissfe.keywords.exportplots:ExportPlotsKeyword",
    "include": "edelweissfe.keywords.include:IncludeKeyword",
}.items():
    _BUILTINS[("keyword", _keywordName.casefold())] = _keywordDotted


#: Resolved-object memo cache: ``(category, name)`` (casefolded) -> ``(target, schema)``. Guarded
#: by ``_lock`` (see :func:`lookup`).
_resolved: dict[tuple[str, str], tuple[Any, type | None]] = {}

#: Guards read-check-write access to ``_resolved``. Chosen strategy (see :func:`lookup`): a plain
#: mutex around "look up the dotted string, import it, store it" rather than a lock-free/idempotent
#: design, because resolving a dotted string can execute arbitrary module-level code (the target
#: module's imports) whose *side effects* -- not just the returned object -- must not race under
#: PYTHON_GIL=0. A dict-level race (two threads both missing the cache and both importing) would
#: itself be harmless (importlib's own per-module lock in ``sys.modules`` already deduplicates the
#: actual import, and re-running ``getattr`` is pure), but without this lock two threads could still
#: both observe a cache miss, both resolve, and then interleave two dict insertions for the *same*
#: key with two (structurally equal but not necessarily identical, e.g. if resolution ever grows a
#: per-call side effect) values. Serializing the whole resolve-and-store makes the outcome
#: independent of scheduling: exactly one thread resolves, every other thread either sees the
#: cached result or waits for the lock and then sees it.
_lock = threading.Lock()

#: Explicit :func:`register` calls: ``(category, name)`` -> ``(target, schema)``. Kept separate from
#: ``_resolved`` (which also holds lazily-memoized *resolutions*) so that whether a registration
#: collides does not depend on whether something happened to look the name up first -- i.e. so
#: collision detection is not itself import-order dependent, which is the pathology this module
#: exists to remove.
_registered: dict[tuple[str, str], tuple[Any, type | None]] = {}

#: Whether :func:`_auditEntryPoints` has already run in this process.
_entryPointsAudited = False


def _resolveDottedString(dotted: str) -> Any:
    """Import ``module.path:AttrName`` and return the attribute.

    Parameters
    ----------
    dotted
        A string of the form ``"module.path:AttrName"``.

    Returns
    -------
    Any
        ``getattr(importlib.import_module(modulePath), attrName)``.
    """
    modulePath, _, attrName = dotted.partition(":")
    module = importlib.import_module(modulePath)
    return getattr(module, attrName)


def _entryPointDottedString(category: str, name: str) -> str | None:
    """Look up a third-party-registered dotted string via ``importlib.metadata`` entry points.

    Entry points are (re-)queried on every call rather than cached at import time or behind a
    one-shot "have we merged yet" flag: ``importlib.metadata.entry_points`` is cheap relative to
    actually importing an implementation module, and re-querying keeps the registry correct if a
    package is installed into the running environment mid-process (e.g. in a test that patches
    ``entry_points``) without needing an explicit cache-invalidation API.

    Parameters
    ----------
    category
        The registry category.
    name
        The name within that category.

    Returns
    -------
    str | None
        The dotted string registered by a third party, or ``None`` if none matches.
    """
    wantedName = f"{category}.{name}".casefold()
    matches = {ep.value for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name.casefold() == wantedName}

    if len(matches) > 1:
        # Every match is collected rather than returning the first, because `entry_points()` has no
        # documented ordering: taking the first made the winner depend on installation order, and
        # differ between environments with the same packages installed.
        raise RegistryConflictError(
            f"Several packages claim the '{category}' name '{name}' via '{ENTRY_POINT_GROUP}' entry "
            f"points: {', '.join(sorted(matches))}. Uninstall one, or have the packages agree on "
            "distinct names -- which of them would win is not defined."
        )

    return matches.pop() if matches else None


def _entryPointsShadowingBuiltins() -> dict[str, tuple[str, str]]:
    """Find entry points claiming a name the built-in table already owns.

    Returns
    -------
    dict[str, tuple[str, str]]
        ``"category.name"`` -> ``(builtinDotted, entryPointDotted)`` for every colliding name.
    """
    shadowed = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        category, _, name = ep.name.casefold().rpartition(".")
        builtinDotted = _BUILTINS.get((category, name))
        if builtinDotted is not None and builtinDotted != ep.value:
            shadowed[ep.name] = (builtinDotted, ep.value)
    return shadowed


def _auditEntryPoints() -> None:
    """Raise if any installed entry point shadows a built-in name.

    :func:`lookup` consults the built-in table *before* entry points, so a third party claiming a
    built-in name would otherwise be silently ignored -- its implementation never loaded, with no
    diagnostic anywhere. That is the mirror image of the :func:`register` hazard and just as quiet.

    Run once per process, from the first :func:`lookup` that misses the memo cache, rather than per
    lookup: the environment does not normally change mid-run, and the audit costs one
    :func:`~importlib.metadata.entry_points` query (a few milliseconds) that would otherwise be
    repeated for every distinct name resolved. The consequence is that the audit reflects the
    environment as of the first lookup; a test that patches ``entry_points`` afterwards still gets
    correct *per-name* resolution (which re-queries every time, see
    :func:`_entryPointDottedString`), just no re-audit.

    Raises
    ------
    RegistryConflictError
        If an entry point claims a name owned by the built-in table.
    """
    global _entryPointsAudited

    if _entryPointsAudited:
        return

    shadowed = _entryPointsShadowingBuiltins()
    if shadowed:
        # Deliberately NOT setting _entryPointsAudited here: a conflict must keep raising on every
        # call until the environment is actually fixed, rather than being permanently silenced by
        # whichever call happens to discover it first.
        details = "; ".join(
            f"'{epName}' is built in as '{builtinDotted}' but an entry point claims it as " f"'{entryPointDotted}'"
            for epName, (builtinDotted, entryPointDotted) in sorted(shadowed.items())
        )
        raise RegistryConflictError(
            f"Installed '{ENTRY_POINT_GROUP}' entry points claim built-in names, which would be "
            f"silently ignored because built-ins resolve first: {details}. Rename the entry "
            "point(s)."
        )

    _entryPointsAudited = True


def availableNames(category: str) -> list[str]:
    """List the names known for ``category`` across both the built-in table and entry points.

    Parameters
    ----------
    category
        The registry category.

    Returns
    -------
    list[str]
        Sorted, de-duplicated list of registered names (original casing where available).
    """
    casefoldedCategory = category.casefold()
    names = {builtinName for (cat, builtinName) in _BUILTINS if cat == casefoldedCategory}
    names |= {registeredName for (cat, registeredName) in _registered if cat == casefoldedCategory}
    prefix = f"{casefoldedCategory}."
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name.casefold().startswith(prefix):
            names.add(ep.name[len(prefix) :])
    return sorted(names)


def isRegistered(category: str, name: str) -> bool:
    """Report whether ``(category, name)`` resolves, *without* importing its implementation.

    :func:`lookup` is the wrong tool for a caller that only wants to validate a name -- it imports
    the implementing module as a side effect, so merely asking "is 'dirichlet' a step action?" would
    pull in every module ever asked about. This is the predicate for those callers; it consults the
    same three sources in the same order, but compares strings only.

    Parameters
    ----------
    category
        The registry category.
    name
        The name within that category.

    Returns
    -------
    bool
        Whether an implementation is registered, by any of the three mechanisms.
    """
    key = (category.casefold(), name.casefold())

    if key in _resolved or key in _BUILTINS:
        return True

    return _entryPointDottedString(*key) is not None


def register(category: str, name: str, target: Any, *, schema: type | None = None, override: bool = False) -> None:
    """Manually register an implementation, bypassing both the built-in table and entry points.

    This is the seam a plugin (or a test) uses to register an object it already holds a reference
    to -- e.g. a synthetic implementation created in-process -- without needing an installed
    entry point. It writes directly into the resolved-object memo cache, so a subsequent
    :func:`lookup` for the same ``(category, name)`` returns ``target`` (and ``schema``)
    immediately, without any import.

    Parameters
    ----------
    category
        The registry category (e.g. ``"outputmanager"``).
    name
        The name to register ``target`` under.
    target
        The class (or factory callable) implementing ``name``.
    schema
        The option schema dataclass associated with ``target``, if any. Usually unnecessary for
        a class that derives from :class:`edelweissfe.utils.schema.OptionSchemaProvider` and
        declares its own ``schema`` attribute -- :func:`lookup` picks that up by itself. Pass it
        explicitly only when ``target`` cannot declare it, e.g. a factory callable or a
        dynamically-created class in a test.
    override
        Replace an existing claim on ``name`` instead of raising. Deliberately explicit: silently
        replacing an implementation is how a built-in could stop existing with no diagnostic, so
        deliberate replacement has to say so.

    Raises
    ------
    RegistryConflictError
        If ``name`` is already claimed in ``category`` -- by the built-in table, or by an earlier
        :func:`register` call with a *different* target -- and ``override`` is not set.
    """
    key = (category.casefold(), name.casefold())

    with _lock:
        if not override:
            previous = _registered.get(key)
            if previous is not None and (previous[0] is not target or previous[1] is not schema):
                raise RegistryConflictError(
                    f"The '{category}' name '{name}' is already registered to {previous[0]!r}; "
                    f"refusing to replace it with {target!r}. Pass override=True to replace it "
                    "deliberately."
                )

            # Checked against the *table of strings*, so this neither imports the built-in nor
            # depends on whether anything resolved it earlier.
            builtinDotted = _BUILTINS.get(key)
            if builtinDotted is not None and previous is None:
                raise RegistryConflictError(
                    f"'{name}' is a built-in '{category}' implementation ('{builtinDotted}'); "
                    f"registering {target!r} under that name would make the built-in unreachable. "
                    "Choose a different name, or pass override=True to shadow it deliberately."
                )

            # A name resolved via an earlier lookup() (built-in table or entry point) is memoized
            # into _resolved directly, without ever touching _registered -- so the checks above miss
            # it. Without this, register() for an already-resolved key silently overwrites the
            # memoized entry a few lines below instead of raising.
            resolvedEntry = _resolved.get(key)
            resolvedDiffersFromTarget = resolvedEntry is not None and (
                resolvedEntry[0] is not target or resolvedEntry[1] is not schema
            )
            if resolvedDiffersFromTarget and previous is None:
                raise RegistryConflictError(
                    f"'{name}' in category '{category}' was already resolved to {resolvedEntry[0]!r} "
                    f"(via lookup); refusing to overwrite it with {target!r}. Pass override=True to "
                    "replace it deliberately."
                )

        _registered[key] = (target, schema)
        _resolved[key] = (target, schema)


def lookup(category: str, name: str) -> tuple[Any, type | None]:
    """Resolve ``(category, name)`` to its implementing object (and option schema, if any).

    Resolution order: the in-process memo cache, then the built-in static table, then
    ``importlib.metadata`` entry points. The result of a successful resolution is memoized, so a
    given ``(category, name)`` is imported at most once per process.

    Thread-safety: guarded by :data:`_lock` using double-checked locking -- the cache is read
    without the lock first (the common case, once warm), and only threads that observe a miss
    contend for the lock, inside which the check is repeated before doing any work. See the
    docstring on :data:`_lock` for why a full mutex was chosen over a lock-free scheme.

    Parameters
    ----------
    category
        The registry category (e.g. ``"outputmanager"``, ``"stepaction"``).
    name
        The name within that category (e.g. ``"ensight"``, ``"dirichlet"``).

    Returns
    -------
    tuple[Any, type | None]
        ``(target, schema)``. Along the dotted-string paths (built-in table and entry points) the
        schema is obtained from the resolved target itself via
        :func:`edelweissfe.utils.schema.schemaOf` -- i.e. the target declares it as a class
        attribute by deriving from
        :class:`edelweissfe.utils.schema.OptionSchemaProvider`. It is ``None`` for targets that
        declare no schema, and structurally always will be for the categories whose targets are
        plain functions rather than classes (see
        :func:`~edelweissfe.utils.schema.schemaOf`). :func:`register` may instead supply a schema
        explicitly, which takes precedence because it writes straight into the memo cache.

    Raises
    ------
    RegistryLookupError
        If no implementation is registered for ``(category, name)``.
    RegistryConflictError
        If two implementations claim ``(category, name)`` -- two entry points, or an entry point
        against a built-in (see :func:`_auditEntryPoints`).
    """
    key = (category.casefold(), name.casefold())

    cached = _resolved.get(key)
    if cached is not None:
        return cached

    with _lock:
        cached = _resolved.get(key)
        if cached is not None:
            return cached

        _auditEntryPoints()

        dotted = _BUILTINS.get(key)
        if dotted is None:
            dotted = _entryPointDottedString(*key)

        if dotted is None:
            known = availableNames(category)
            hint = ""
            if known:
                try:
                    similar = findSimilarString(name, known)
                    hint = f" Did you mean '{similar}'?"
                except ValueError:
                    pass
                message = (
                    f"No '{category}' implementation registered under the name '{name}'. "
                    f"Available: {', '.join(known)}.{hint}"
                )
            else:
                message = f"No '{category}' implementation registered under the name '{name}' " "(no names known)."
            raise RegistryLookupError(message)

        target = _resolveDottedString(dotted)
        result = (target, schemaOf(target))
        _resolved[key] = result
        return result
