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
"""``NID`` under contact: reusing the mass and the damping across a connectivity-only rebuild.

A contact constraint changes its connectivity as the bodies approach and slide, and the parent
solver builds a new discrete system every time it does. Reassembling the mass and the damping on
each of those rebuilds would be a full element loop per increment for operators that cannot have
changed -- no node moved, no element changed -- so ``NID`` copies the mass onto the sparsity pattern of the new
system and keeps the damping instead (``_reuseMassAndDamping``). What has to hold:

* **It happens.** On a deck whose contact connectivity really changes mid-step, the operators are
  reused on those rebuilds, and the sparsity pattern really differs between the two systems -- otherwise the reuse
  was never the non-trivial case.
* **It changes nothing.** The run is bit-for-bit identical to the same run with every rebuild
  forced to reassemble. The reused operators are the reassembled ones, so anything short of
  identity is a bug, not round-off.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

#: Every test of this module runs with each way of summing the system matrix (see ``conftest.py``).
pytestmark = pytest.mark.usefixtures("systemMatrixAssemblyOfNID")

_TRUE_REUSE = NonlinearImplicitDynamic._reuseMassAndDamping
_TRUE_ASSEMBLE = NonlinearImplicitDynamic._assembleMassAndDamping

_DECK = """
*material, name=linearelastic, id=mat
18000, 0.0, 1.0e-3

*section, name=sec, material=mat, type=solid
lower_all
upper_all

*modelGenerator, generator=boxGen, name=lower
nX=2
nY=2
nZ=1
lX=1
lY=1
lZ=1
elType=C3D8

*modelGenerator, generator=boxGen, name=upper
nX=1
nY=1
nZ=1
x0=0.2
y0=0.2
z0=1.02
lX=0.6
lY=0.6
lZ=0.5
elType=C3D8

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface=upper_back
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface=lower_front
name=masterSurf

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=linear,
searchDistance=2.0, sliding=finite

*job, name=nidcontact, domain=3d
*solver, solver=NID, name=theSolver

*step, solver=theSolver
maxInc=0.05, startInc=0.05, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=0.1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pushUpper, nSet=upper_front, field=displacement, 1=0.4, 2=0.0, 3=-0.05, f(t)='t*t*(3-2*t)'
"""


def _run(tmp_path, name: str, monkeypatch, forceReassembly: bool):
    record = {"reused": [], "assembled": 0}

    def reuse(self, system):
        if forceReassembly:
            return None
        result = _TRUE_REUSE(self, system)
        if result is not None:
            pattern = self._K.csrMatrix
            record["reused"].append((system.M.indptr, system.M.indices, pattern.indptr, pattern.indices))
        return result

    def assemble(self, model, dynamicDofs, dynamicFields, lumpedMassOfEntities):
        record["assembled"] += 1
        return _TRUE_ASSEMBLE(self, model, dynamicDofs, dynamicFields, lumpedMassOfEntities)

    monkeypatch.setattr(NonlinearImplicitDynamic, "_reuseMassAndDamping", reuse)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_assembleMassAndDamping", assemble)

    path = tmp_path / f"{name}.inp"
    path.write_text(_DECK)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    field = model.nodeFields["displacement"]
    return record, [np.array(field[entry]) for entry in "UVA"], model


def test_mass_and_damping_are_reused_across_a_contact_connectivity_change(tmp_path, monkeypatch):
    record, _, model = _run(tmp_path, "reused", monkeypatch, forceReassembly=False)

    assert record["reused"], "no connectivity-only rebuild happened; the deck exercises nothing"
    assert record["assembled"] == 1, "reassembled {:} times".format(record["assembled"])
    assert any(
        not (np.array_equal(indptrOld, indptrNew) and np.array_equal(indicesOld, indicesNew))
        for indptrOld, indicesOld, indptrNew, indicesNew in record["reused"]
    ), "the sparsity pattern never changed, so the reuse was always trivial"
    assert model.time > 0.0


def test_reusing_is_bit_identical_to_reassembling(tmp_path, monkeypatch):
    reused, reusedState, _ = _run(tmp_path, "reused", monkeypatch, forceReassembly=False)
    reassembled, reassembledState, _ = _run(tmp_path, "reassembled", monkeypatch, forceReassembly=True)

    assert reused["reused"] and reassembled["assembled"] > 1

    for c, r, name in zip(reusedState, reassembledState, "UVA"):
        np.testing.assert_array_equal(c, r, err_msg=name)


@pytest.mark.parametrize("change", ["numbering", "elements"])
def test_the_mass_is_reassembled_when_the_numbering_or_the_elements_differ(change):
    """The reuse compares the digest of the dof numbering and the set of elements of the two systems, and refuses as
    soon as either differs; the stiffness is never asked for a copy then."""

    def system(digest, elements):
        return SimpleNamespace(
            numbering=SimpleNamespace(digest=digest),
            locations=SimpleNamespace(elements=list(elements)),
        )

    old = system("a", ["e1", "e2"])
    new = system("b", ["e1", "e2"]) if change == "numbering" else system("a", ["e1", "e3"])
    solver = SimpleNamespace(system=new, _K=None)

    assert NonlinearImplicitDynamic._reuseMassAndDamping(solver, SimpleNamespace(system=old)) is None
