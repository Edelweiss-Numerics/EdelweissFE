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
"""``NID`` under contact: keeping the mass and the damping across new locations.

A contact constraint changes its connectivity as the bodies approach and slide, and the solver gets new locations on the
same numbering every time it does. Reassembling the mass and the damping on each of those would be a full element loop
per increment for operators that cannot have changed -- no node moved, no element changed -- so ``NID`` copies the mass
onto the sparsity pattern of the new system matrix and keeps the damping instead
(:meth:`~edelweissfe.solvers.nonlinearimplicitdynamic.NonlinearImplicitDynamic.onNewLocations`). What has to hold:

* **It happens.** On a deck whose contact connectivity really changes mid-step, the mass is moved on those new
  locations, and the sparsity pattern really differs between the two systems -- otherwise moving it was never the
  non-trivial case.
* **It changes nothing.** The run is bit-for-bit identical to the same run with the mass reassembled on every new
  locations. The moved mass is the reassembled one, so anything short of identity is a bug, not round-off.
* **It is on the new pattern.** The moved mass has exactly the sparsity pattern of the new system matrix.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

#: Every test of this module runs with each way of summing the system matrix (see ``conftest.py``).
pytestmark = pytest.mark.usefixtures("systemMatrixAssemblyOfNID")

_TRUE_ON_NEW_LOCATIONS = NonlinearImplicitDynamic.onNewLocations
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
    record = {"moved": [], "assembled": 0, "onTheNewPattern": []}

    def onNewLocations(self):
        old = self._newmarkSystem
        _TRUE_ON_NEW_LOCATIONS(self)
        if old is not None and self._newmarkSystem is not None:
            pattern = self._K.csrMatrix
            record["moved"].append((old.M.indptr, old.M.indices, pattern.indptr, pattern.indices))
            moved = self._newmarkSystem.M
            record["onTheNewPattern"].append(
                np.array_equal(moved.indptr, pattern.indptr) and np.array_equal(moved.indices, pattern.indices)
            )
        if forceReassembly:
            self._newmarkSystem = None

    def assemble(self, model, dynamicDofs, dynamicFields, lumpedMassOfEntities):
        record["assembled"] += 1
        return _TRUE_ASSEMBLE(self, model, dynamicDofs, dynamicFields, lumpedMassOfEntities)

    monkeypatch.setattr(NonlinearImplicitDynamic, "onNewLocations", onNewLocations)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_assembleMassAndDamping", assemble)

    path = tmp_path / f"{name}.inp"
    path.write_text(_DECK)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    field = model.nodeFields["displacement"]
    return record, [np.array(field[entry]) for entry in "UVA"], model


def test_mass_and_damping_are_kept_across_a_contact_connectivity_change(tmp_path, monkeypatch):
    record, _, model = _run(tmp_path, "kept", monkeypatch, forceReassembly=False)

    assert record["moved"], "no change of the locations happened on the numbering; the deck exercises nothing"
    assert record["assembled"] == 1, "reassembled {:} times".format(record["assembled"])
    assert any(
        not (np.array_equal(indptrOld, indptrNew) and np.array_equal(indicesOld, indicesNew))
        for indptrOld, indicesOld, indptrNew, indicesNew in record["moved"]
    ), "the sparsity pattern never changed, so moving the mass was always trivial"
    assert all(record["onTheNewPattern"])
    assert model.time > 0.0


def test_moving_the_mass_is_bit_identical_to_reassembling_it(tmp_path, monkeypatch):
    kept, keptState, _ = _run(tmp_path, "kept", monkeypatch, forceReassembly=False)
    reassembled, reassembledState, _ = _run(tmp_path, "reassembled", monkeypatch, forceReassembly=True)

    assert kept["moved"] and reassembled["assembled"] > 1

    for c, r, name in zip(keptState, reassembledState, "UVA"):
        np.testing.assert_array_equal(c, r, err_msg=name)
