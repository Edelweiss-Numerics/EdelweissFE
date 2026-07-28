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

"""L3 lazy registry for the input-language redesign (see ``PLAN_INPUT_SYSTEM.md``, P1).

Maps ``(category, name)`` to the implementing object (a class, or occasionally a factory
function -- see the coverage notes below) that EdelweissFE's own modules or third-party packages
(EdelweissMeshfree, plugins) provide for that name.

Four properties are load-bearing; the first three are each covered by a dedicated test in
``tests/test_registry.py``:

1. **Zero eager imports.** Importing this module must not import any element, material, solver,
   or output-manager module -- it only ever imports :mod:`importlib` and
   :mod:`importlib.metadata`. Resolving a name (importing the module that actually implements it)
   happens exclusively inside :func:`lookup`, on first use of that particular ``(category,
   name)`` pair.
2. **Built-ins work with a stale editable install.** Entry points declared in a package's
   ``pyproject.toml`` only materialize in ``importlib.metadata`` after ``pip install -e .`` is
   re-run -- so a registry that relies solely on entry-point discovery would leave a fresh
   checkout of EdelweissFE itself broken until someone remembers to reinstall it. To avoid that,
   EdelweissFE's own modules are additionally listed in the static ``_BUILTINS`` table below as
   plain ``"module.path:AttrName"`` strings -- ordinary Python string literals, requiring no
   metadata regeneration whatsoever -- and entry points are layered *on top* of that table for
   third-party discovery.
3. **Thread-safe memoization.** ``lookup()`` caches the resolved object in ``_resolved`` so a
   given ``(category, name)`` is only imported once. Under free-threading (``PYTHON_GIL=0``),
   multiple threads can call ``lookup()`` for the same key concurrently; see the docstring of
   :func:`lookup` for the chosen strategy.
4. **Names are case-insensitive, deliberately, and that belongs here rather than in the input-file
   front-end.** Both ``category`` and ``name`` are casefolded on the way in (:func:`lookup`,
   :func:`register`) and stored casefolded in ``_BUILTINS``. This is not a convenience shortcut and
   should not be "corrected" to exact matching: the whole point of this registry is that an
   external package (EdelweissMeshfree, a plugin) reaches these modules *without* the ``.inp``
   parser in the loop, so if case-insensitivity lived only in the parser, the same name would
   resolve differently depending on which front-end it arrived through. It is also what 12 of the
   13 ``config/*.py`` registries this replaces already did via ``name.lower()``; the lone
   exception, ``config/solvers.py``, is case-sensitive, so solver names becoming case-insensitive
   is this registry's one behavioral change (strictly more permissive -- no existing ``.inp``
   changes meaning). See rule (c) in ``PLAN_INPUT_SYSTEM.md`` §3, amended to say so. Note the
   consequence tracked as a P2 deliverable: casefolded keys widen the window for registration
   collisions, and :func:`register` does not yet detect them.

Built-in coverage
------------------
The ``_BUILTINS`` table below currently covers these categories, enumerated by hand from the
corresponding ``edelweissfe`` subpackage (see the module docstring history / this branch's report
for exactly how each list was derived):

``outputmanager``, ``section``, ``constraint``, ``stepaction``, ``generator``,
``analyticalfield``, ``solver``, ``step``, ``modelmodifier``, ``statetransferstrategy``.

It deliberately does **not** yet cover:

- ``element`` and ``material`` -- ``config/elementlibrary.py`` and ``config/materiallibrary.py``
  dispatch on a *second* axis (``provider``: ``edelweiss`` vs. ``marmot`` vs.
  ``marmotsingleqpelement``) that changes the lookup semantics entirely (e.g. ``marmot`` ignores
  ``name`` and always returns the same wrapper class). Folding that into a flat ``(category,
  name)`` registry requires a design decision (does ``name`` encode the provider too?) that is out
  of scope for P1 and left to P4, alongside the other 11 ``config/*.py`` registries.
- ``linsolver`` -- ``config/linsolve.py``'s entries are not uniformly "a dotted string to a plain
  class/callable": ``superlu``/``umfpack`` are inline ``lambda`` closures with no module-level
  name to point a dotted string at, and ``gmres``/``amgcl`` require constructing a wrapper object
  with call-site-specific options before the usable callable exists. Left to P4.

This is a genuine subset, not full coverage dressed up -- do not extend call sites to assume every
category is present until P4 actually finishes the fold-in.
"""

from __future__ import annotations

import importlib
import threading
from importlib.metadata import entry_points
from typing import Any

from edelweissfe.utils.misc import findSimilarString

#: Entry-point group name third-party packages register EdelweissFE-discoverable implementations
#: under, e.g. in a plugin's ``pyproject.toml``::
#:
#:     [project.entry-points."edelweissfe.plugins"]
#:     "outputmanager.myoutputmanager" = "mypackage.mymodule:MyOutputManager"
ENTRY_POINT_GROUP = "edelweissfe.plugins"


class RegistryLookupError(LookupError):
    """Raised by :func:`lookup` when no implementation is registered for a requested name.

    Unlike the legacy ``InputLanguage``/``findSimilarString`` combination (which raises a bare
    ``Exception`` with the message "You tried to find a string similar to ... in an empty list."
    when the candidate list happens to be empty), this always produces a message naming the
    category and, whenever at least one name is registered for it, a "did you mean" suggestion.
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
    "generateModelData",
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

# solver / step / modelmodifier / statetransferstrategy are not "one module per name" -- copied
# by hand from config/solvers.py's solverLibrary, config/steps.py's stepLibrary,
# modelmodifiers/adaptivity, and config/statetransferstrategies.py's _STRATEGIES respectively.
# Kept in sync manually for now; P4 removes the duplication when config/*.py folds into this file.
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
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name.casefold() == wantedName:
            return ep.value
    return None


def _availableNames(category: str) -> list[str]:
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
    prefix = f"{casefoldedCategory}."
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name.casefold().startswith(prefix):
            names.add(ep.name[len(prefix) :])
    return sorted(names)


def register(category: str, name: str, target: Any, *, schema: type | None = None) -> None:
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
        The L2 option schema dataclass associated with ``target``, if any. ``None`` for modules
        that have not yet been given a schema (true of every built-in entry as of P1 -- P2 wires
        schemas in per module).
    """
    key = (category.casefold(), name.casefold())
    with _lock:
        _resolved[key] = (target, schema)


def lookup(category: str, name: str) -> tuple[Any, type | None]:
    """Resolve ``(category, name)`` to its implementing object (and L2 schema, if any).

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
        ``(target, schema)``. ``schema`` is ``None`` for every entry resolved via the built-in
        table or a plain entry point as of P1 (no module has an L2 schema wired in yet); P2+
        registrations made via :func:`register` may supply one.

    Raises
    ------
    RegistryLookupError
        If no implementation is registered for ``(category, name)``.
    """
    key = (category.casefold(), name.casefold())

    cached = _resolved.get(key)
    if cached is not None:
        return cached

    with _lock:
        cached = _resolved.get(key)
        if cached is not None:
            return cached

        dotted = _BUILTINS.get(key)
        if dotted is None:
            dotted = _entryPointDottedString(*key)

        if dotted is None:
            availableNames = _availableNames(category)
            hint = ""
            if availableNames:
                try:
                    similar = findSimilarString(name, availableNames)
                    hint = f" Did you mean '{similar}'?"
                except ValueError:
                    pass
                message = (
                    f"No '{category}' implementation registered under the name '{name}'. "
                    f"Available: {', '.join(availableNames)}.{hint}"
                )
            else:
                message = f"No '{category}' implementation registered under the name '{name}' " "(no names known)."
            raise RegistryLookupError(message)

        target = _resolveDottedString(dotted)
        result = (target, None)
        _resolved[key] = result
        return result
