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
"""``NID`` under contact: carrying the mass and the damping over a connectivity-only rebuild.

A contact constraint changes its connectivity as the bodies approach and slide, and the parent
solver builds a new equation system every time it does. Reassembling the mass and the damping on
each of those rebuilds would be a full element loop per increment for operators that cannot have
changed -- no node moved, no element changed -- so ``NID`` carries them over into the new layout
instead (``_carryOperatorsOver``). What has to hold:

* **It happens.** On a deck whose contact connectivity really changes mid-step, the operators are
  carried over on those rebuilds, and the constraint part of the layout really differs between the
  two managers -- otherwise the carry-over was never the non-trivial case.
* **It changes nothing.** The run is bit-for-bit identical to the same run with every rebuild
  forced to reassemble. The carried-over operators are the reassembled ones, so anything short of
  identity is a bug, not round-off.
"""

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

_TRUE_CARRY_OVER = NonlinearImplicitDynamic._carryOperatorsOver
_TRUE_ASSEMBLE = NonlinearImplicitDynamic._assembleOperators

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
    record = {"carriedOver": [], "assembled": 0}

    def carryOver(self, system, model):
        if forceReassembly:
            return None
        result = _TRUE_CARRY_OVER(self, system, model)
        if result is not None:
            record["carriedOver"].append(
                (len(system.dofManager.I), len(self.theDofManager.I), system.dofManager.I, self.theDofManager.I)
            )
        return result

    def assemble(self, model, dynamicDofs, dynamicFields):
        record["assembled"] += 1
        return _TRUE_ASSEMBLE(self, model, dynamicDofs, dynamicFields)

    monkeypatch.setattr(NonlinearImplicitDynamic, "_carryOperatorsOver", carryOver)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_assembleOperators", assemble)

    path = tmp_path / f"{name}.inp"
    path.write_text(_DECK)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    field = model.nodeFields["displacement"]
    return record, [np.array(field[entry]) for entry in "UVA"], model


def test_operators_are_carried_over_a_contact_connectivity_change(tmp_path, monkeypatch):
    record, _, model = _run(tmp_path, "carried", monkeypatch, forceReassembly=False)

    assert record["carriedOver"], "no connectivity-only rebuild happened; the deck exercises nothing"
    assert record["assembled"] == 1, "reassembled {:} times".format(record["assembled"])
    assert any(
        nOld != nNew or not np.array_equal(IOld, INew) for nOld, nNew, IOld, INew in record["carriedOver"]
    ), "the constraint part of the layout never changed, so the carry-over was always trivial"
    assert model.time > 0.0


def test_carrying_over_is_bit_identical_to_reassembling(tmp_path, monkeypatch):
    carried, carriedState, _ = _run(tmp_path, "carried", monkeypatch, forceReassembly=False)
    reassembled, reassembledState, _ = _run(tmp_path, "reassembled", monkeypatch, forceReassembly=True)

    assert carried["carriedOver"] and reassembled["assembled"] > 1

    for c, r, name in zip(carriedState, reassembledState, "UVA"):
        np.testing.assert_array_equal(c, r, err_msg=name)
