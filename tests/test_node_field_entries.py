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
"""Every node field holds the entries U, dU, P, V and A, from its construction on, and no others (see the
documentation page "Node field results"). A DofManager serves exactly these entries of exactly the node fields it
numbers, and a checkpoint holds exactly these entries of every node field."""

import h5py
import numpy as np
import pytest
from test_femodel_restart import _buildSingleElementModel

from edelweissfe.fields.nodefield import NodeField
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.checkpoint import CHECKPOINT_FORMAT_VERSION, ResumeCheckpoint
from edelweissfe.utils.exceptions import RestartError
from edelweissfe.variables.fieldvariable import FieldVariable

_FIELD = "displacement"


def _nodes(count, field=_FIELD):
    nodes = [Node(label, np.array([float(label), 0.0])) for label in range(1, count + 1)]
    for node in nodes:
        node.fields[field] = FieldVariable(node, field)
    return nodes


def test_a_node_field_holds_every_entry_from_its_construction_on():
    nodes = _nodes(3)
    nodeField = NodeField(_FIELD, 2, NodeSet("all", nodes))

    assert NodeField.ENTRIES == ("U", "dU", "P", "V", "A")
    for entry in NodeField.ENTRIES:
        assert nodeField[entry].shape == (3, 2)
        assert not np.any(nodeField[entry])

    nodeField.resize(nodes + _nodes(4)[3:])
    for entry in NodeField.ENTRIES:
        assert nodeField[entry].shape == (4, 2)


def test_a_dof_manager_refuses_an_entry_no_node_field_holds():
    nodeField = NodeField(_FIELD, 2, NodeSet("all", _nodes(2)))
    dofManager = DofManager([nodeField])
    vector = dofManager.constructDofVector()

    with pytest.raises(KeyError, match="'Q'"):
        dofManager.writeDofVectorToNodeField(vector, nodeField, "Q")
    with pytest.raises(KeyError, match="'Q'"):
        dofManager.writeNodeFieldToDofVector(vector, nodeField, "Q")


def test_a_dof_manager_reads_and_writes_as_a_whole_only_the_node_fields_it_numbers():
    nodes = _nodes(3)
    numbered = NodeField(_FIELD, 2, NodeSet("numbered", nodes[:2]))
    ofTheModel = NodeField(_FIELD, 2, NodeSet("all", nodes))
    dofManager = DofManager([numbered])
    vector = dofManager.constructDofVector()

    # the node field of the model has the name of the numbered one, but other rows
    with pytest.raises(ValueError, match="another node field"):
        dofManager.writeDofVectorToNodeField(vector, ofTheModel, "U")
    with pytest.raises(ValueError, match="another node field"):
        dofManager.writeNodeFieldToDofVector(vector, ofTheModel, "U")

    rotationNodes = _nodes(3, "rotation")
    otherField = NodeField("rotation", 2, NodeSet("all", rotationNodes))
    with pytest.raises(ValueError, match="does not number"):
        dofManager.writeNodeFieldToDofVector(vector, otherField, "U", NodeSet("all", rotationNodes))

    # on a node set, the rows are found by node: the node field of the model is read where it is numbered
    ofTheModel["U"][:] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    dofManager.writeNodeFieldToDofVector(vector, ofTheModel, "U", NodeSet("all", nodes))
    np.testing.assert_array_equal(np.asarray(vector), [1.0, 2.0, 3.0, 4.0])

    dofManager.writeDofVectorToNodeField(vector, numbered, "dU")
    np.testing.assert_array_equal(numbered["dU"], [[1.0, 2.0], [3.0, 4.0]])


def test_a_checkpoint_holds_every_entry_of_every_node_field(tmp_path):
    model, _ = _buildSingleElementModel()
    for index, entry in enumerate(NodeField.ENTRIES):
        model.nodeFields[_FIELD][entry][:] = index + 1.0

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        model.writeRestart(f)
    with h5py.File(checkpointPath, "r") as f:
        assert set(f["nodeFields"][_FIELD].keys()) == set(NodeField.ENTRIES)

    for entry in NodeField.ENTRIES:
        model.nodeFields[_FIELD][entry][:] = 0.0
    with h5py.File(checkpointPath, "r") as f:
        model.readRestart(f)
    for index, entry in enumerate(NodeField.ENTRIES):
        assert np.all(model.nodeFields[_FIELD][entry] == index + 1.0)


def test_a_checkpoint_lacking_an_entry_is_refused_before_anything_is_restored(tmp_path):
    model, _ = _buildSingleElementModel()
    model.time = 2.0

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        model.writeRestart(f)
        del f["nodeFields"][_FIELD]["V"]

    model.time = 1.0
    with h5py.File(checkpointPath, "r") as f:
        with pytest.raises(RestartError, match="entries"):
            model.readRestart(f)
    assert model.time == 1.0


def test_a_checkpoint_of_an_earlier_format_is_refused(tmp_path):
    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        f.attrs["formatVersion"] = CHECKPOINT_FORMAT_VERSION - 1

    with pytest.raises(RestartError, match="format version {:}".format(CHECKPOINT_FORMAT_VERSION - 1)):
        ResumeCheckpoint(str(checkpointPath))
