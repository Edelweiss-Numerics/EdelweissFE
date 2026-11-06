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
Bond-slip coupling of embedded reinforcement bars to host continuum elements.

The generator splits every bar (truss) element of ``rebarElSet`` at the boundaries of the host elements of
``hostElSet`` and creates one Marmot ``EmbeddedBondElement`` per part. Its nodes are the nodes of the bar element
followed by the nodes of the host element; the bar keeps its own, independent displacement field, and the bond
element transfers the bond stress of a pluggable bond-slip law (e.g., ``LINEARELASTICBONDSLIP``,
``MODELCODE2010BONDSLIP``) between bar and host. The bar and host meshes are independent of each other.

Parts of a bar lying outside of all host elements (e.g., a free end of a pull-out specimen) remain unbonded.

The bars may also be beam elements (e.g., Marmot ``B2D2``, ``B3D3``, reported as ``bar2``/``bar3``): the bond elements couple
only their displacement field, the beam rotations are not coupled to the host (as for ``rotations=free`` of the
embedded region constraint), and the bond sees the linear interpolation of the beam axis between its nodes, not the
beam's cubic deflection -- use ``splitBars=True`` (or a beam mesh finer than the host) for bent beams. Split beams
get the type and the section of the original beam.

With ``splitBars=True``, every bar element crossing host element boundaries is first replaced by bar elements
(of the same type) with nodes at the crossings, so that each bar element lies in exactly one host element. Without
splitting, a stiff bond forces the host displacement along a bar element to follow the bar element's (linear)
interpolation also between its nodes, which over-constrains (locks) the host for stiff bond laws, in particular in
the normal direction. With splitting, a stiff bond converges to the embedded region constraint (perfect bond) for
the split bar mesh.

The generator needs the bar and host elements, so it must run after the manual mesh generation
(``executeAfterManualGeneration=True``). It creates the element set ``<name>_bond`` and a section assigning the
bond-slip law and the bond element properties (perimeter, part of the bar element, integration points).

With ``largeSlip=True`` (and ``maxSlip``), Marmot ``EmbeddedLargeSlipBondElement`` elements are created instead: the bond
is integrated over the channel (the original bar path in each host element, carrying the bond history), and the
bar partner of every channel point is searched among the bar elements within ``maxSlip`` of the channel. The bars
may then slide far: the part of a bar pulled out of the host is unbonded and the bonded length shrinks. The bar
elements are chained into polylines for this, so they must form unbranched chains (truss or beam elements, linear or
quadratic, in any orientation). Each bond element couples a host element to a window of consecutive bar elements
(``EBLS<nDim>D<nBarNodes><host>W<n>``, at most 48); a slip beyond its window stops the analysis with an error, so
``maxSlip`` must not be smaller than the largest slip expected.

For perfect bond, use the ``embeddedRegion`` constraint (:mod:`edelweissfe.constraints.embeddedregion`) instead, or a
stiff ``LINEARELASTICBONDSLIP`` law.

.. code-block:: edelweiss
    :caption: Example:

    *material, name=ModelCode2010BondSlip, id=bond, provider=marmot
        **  tauMax, s1,  s2,  s3,   alpha, tauF, K0,    Kn
            13.7,   1.0, 2.0, 10.0, 0.4,   5.5,  200.0, 1000.0

    *modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
        rebarElSet=rebars
        hostElSet=concrete
        material=bond
        perimeter=37.7
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.embedding import (
    MIN_PART_LENGTH,
    HostElementLocator,
    barGeometry,
    barLength,
    barPoint,
    splitBarElementAtHostBoundaries,
)
from edelweissfe.utils.schema import schemaField

#: The largest window of bar elements of the registered Marmot large-slip bond elements.
MAX_LARGE_SLIP_WINDOW = 48

#: The abbreviation of a host element shape in the Marmot bond element names.
HOST_SHAPE_CODES = {"quad4": "Q4", "quad8": "Q8", "tetra4": "T4", "hexa8": "H8", "hexa20": "H20"}


def bondElementType(nDim: int, barShape: str, hostShape: str) -> str:
    """The name of the Marmot bond element, ``EB<nDim>D<number of bar nodes><host shape>``, e.g., ``EB2D2Q4``."""
    try:
        return f"EB{nDim}D{2 if barShape == 'bar2' else 3}{HOST_SHAPE_CODES[hostShape]}"
    except KeyError:
        raise NotImplementedError(f"No bond element for '{hostShape}' host elements.")


@dataclass(frozen=True)
class EmbeddedBondSchema:
    """The options this generator accepts, owned by this module and never mutated from outside it."""

    rebarElSet: str | None = schemaField(
        description="The element set of the reinforcement bar (truss or beam) elements.",
        dtype=str,
        default=None,
        required=True,
    )
    hostElSet: str | None = schemaField(
        description="The element set of the host continuum elements.", dtype=str, default=None, required=True
    )
    material: str | None = schemaField(
        description="The id of the (Marmot) bond-slip law, defined by *material.",
        dtype=str,
        default=None,
        required=True,
    )
    perimeter: float | None = schemaField(
        description="The perimeter of the bars, i.e., the bond surface per bar length.",
        dtype=float,
        default=None,
        required=True,
    )
    nIntegrationPoints: int = schemaField(
        description="The number of Gauss-Lobatto points per bond element (2 to 5); 0 for the default of the element "
        "(2 for linear, 3 for quadratic bars).",
        dtype=int,
        default=0,
    )
    tolerance: float = schemaField(
        description="Tolerance of the parametric coordinates on the boundary of a host element.",
        dtype=float,
        default=1e-10,
    )
    largeSlip: bool = schemaField(
        description="Large slip: the bars may slide out of (or into) their hosts, see the module documentation. "
        "Requires maxSlip.",
        dtype=bool,
        default=False,
    )
    maxSlip: float | None = schemaField(
        description="Large slip only: the largest slip expected, which sets the window of bar elements each bond "
        "element couples to; a larger slip stops the analysis with an error.",
        dtype=float,
        default=None,
    )
    splitBars: bool = schemaField(
        description="Replace the bar elements by bar elements with nodes at the crossings of the host element "
        "boundaries, such that every bar element lies in one host element.",
        dtype=bool,
        default=False,
    )


class EmbeddedBondSection:
    """Assigns the bond-slip law and the per-element properties to the generated bond elements.

    Unlike the sections of the input file, the element properties differ from element to element (the part of the
    bar element they cover), so this section is created by the generator itself and registered in
    ``model.sections``; it is applied with all other sections in :meth:`~edelweissfe.models.femodel.FEModel.prepareYourself`.
    """

    def __init__(self, name: str, material: dict, elementSet: ElementSet, properties: dict):
        self.name = name
        self.material = material
        self.elSets = [elementSet]
        self.properties = properties

    def assignSectionPropertiesToModel(self, model):
        for el in self.elSets[0]:
            self.assignSectionToElement(el, model)
        return model

    def assignSectionToElement(self, element, model):
        self.assignSectionPropertiesToElement(element)

    def assignSectionPropertiesToElement(self, element, material: dict = None):
        """Assign the per-element properties and the (possibly modified, e.g., by a ``changeMaterialProperty``
        step action) bond-slip law to a bond element."""
        material = material or self.material
        element.setProperties(self.properties[element.elNumber])
        element.initializeElement()
        element.setMaterial(material["name"], material["properties"])


class Generator(GeneratorBase):
    """Creates bond-slip elements between embedded reinforcement bars and host continuum elements."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = EmbeddedBondSchema

    def __init__(
        self, name: str, model: FEModel, journal: Journal, *, configuration: EmbeddedBondSchema = EmbeddedBondSchema()
    ):
        if configuration.rebarElSet not in model.elementSets or configuration.hostElSet not in model.elementSets:
            raise ValueError(
                f"{name}: element sets '{configuration.rebarElSet}' and '{configuration.hostElSet}' must exist; "
                "run this generator with executeAfterManualGeneration=True."
            )
        material = model.materials[configuration.material]
        if not isinstance(material, dict):
            raise ValueError(f"{name}: the bond-slip law '{configuration.material}' must be a Marmot material.")

        rebarElements = list(model.elementSets[configuration.rebarElSet])
        hostElements = list(model.elementSets[configuration.hostElSet])
        locator = HostElementLocator(hostElements, tolerance=configuration.tolerance)
        nDim = model.domainSize

        if configuration.splitBars:
            rebarElements = [
                newBar for bar in rebarElements for newBar in _splitBarAtHostBoundaries(bar, locator, model)
            ]
            model._populateNodeFieldVariablesFromElements()

        bondElements = []
        properties = {}
        bondedLength, unbondedLength = 0.0, 0.0

        def createBondElement(elType, nodes, props):
            (label,) = model.topology.reserveElementNumbers(1)
            bondElement = getElementClass(elType, "marmot")(elType, label)
            bondElement.setNodes(nodes)
            model.createElement(bondElement)
            bondElements.append(bondElement)
            if configuration.nIntegrationPoints:
                props = props + [float(configuration.nIntegrationPoints)]
            properties[label] = np.array(props, dtype=float)

        def checkIndependent(barNodes, host):
            shared = set(barNodes) & set(host.nodes)
            if shared:
                raise ValueError(
                    f"{name}: bar node(s) {sorted(n.label for n in shared)} are nodes of host element "
                    f"{host.elNumber}; bond-slip needs bar nodes independent of the host (use the embeddedRegion "
                    "constraint for perfect bond of a conforming bar mesh)."
                )

        if configuration.largeSlip:
            if not configuration.maxSlip or configuration.maxSlip <= 0:
                raise ValueError(f"{name}: largeSlip requires a positive maxSlip.")
            for chain in _chainBars(rebarElements):
                barShape = chain.shape
                for k, (bar, reversed_) in enumerate(chain.elements):
                    barShapeK, barCoordinates = barGeometry(bar)
                    parts, _ = splitBarElementAtHostBoundaries(bar, locator)
                    length = barLength(barShapeK, barCoordinates)
                    partsLength = sum(barLength(barShapeK, barCoordinates, p.etaStart, p.etaEnd) for p in parts)
                    bondedLength += partsLength
                    unbondedLength += length - partsLength
                    for part in parts:
                        # the channel in the chain parameter, c = k + (xi + 1) / 2 with xi along the chain
                        xis = sorted((-part.etaEnd, -part.etaStart) if reversed_ else (part.etaStart, part.etaEnd))
                        cStart, cEnd = (k + 0.5 * (xi + 1) for xi in xis)
                        SStart = chain.S[k] + (cStart - k) * chain.lengths[k]
                        SEnd = chain.S[k] + (cEnd - k) * chain.lengths[k]
                        k0 = max(i for i in range(k + 1) if chain.S[i] <= SStart - configuration.maxSlip or i == 0)
                        k1 = min(
                            i
                            for i in range(k, len(chain.elements))
                            if chain.S[i + 1] >= SEnd + configuration.maxSlip or i == len(chain.elements) - 1
                        )
                        window = k1 - k0 + 1
                        if window > MAX_LARGE_SLIP_WINDOW:
                            raise ValueError(
                                f"{name}: maxSlip={configuration.maxSlip} needs a window of {window} bar elements, at "
                                f"most {MAX_LARGE_SLIP_WINDOW} are supported; refine less or reduce maxSlip."
                            )
                        windowNodes = chain.windowNodes(k0, k1)
                        checkIndependent(windowNodes, part.hostElement)
                        elType = (
                            "EBLS" + bondElementType(nDim, barShape, part.hostElement.ensightType)[2:] + f"W{window}"
                        )
                        createBondElement(
                            elType,
                            list(part.hostElement.nodes) + windowNodes,
                            [
                                configuration.perimeter,
                                cStart - k0,
                                cEnd - k0,
                                float(k0 == 0),
                                float(k1 == len(chain.elements) - 1),
                            ],
                        )
        else:
            for bar in rebarElements:
                barShape, barCoordinates = barGeometry(bar)
                parts, _ = splitBarElementAtHostBoundaries(bar, locator)
                length = barLength(barShape, barCoordinates)
                partsLength = sum(barLength(barShape, barCoordinates, p.etaStart, p.etaEnd) for p in parts)
                bondedLength += partsLength
                unbondedLength += length - partsLength

                for part in parts:
                    checkIndependent(bar.nodes, part.hostElement)
                    createBondElement(
                        bondElementType(nDim, barShape, part.hostElement.ensightType),
                        list(bar.nodes) + list(part.hostElement.nodes),
                        [configuration.perimeter, part.etaStart, part.etaEnd],
                    )

        elementSet = ElementSet(f"{name}_bond", bondElements)
        model.elementSets[elementSet.name] = elementSet
        model.sections[f"{name}_section"] = EmbeddedBondSection(f"{name}_section", material, elementSet, properties)

        journal.message(
            f"{len(bondElements)} bond elements for {len(rebarElements)} bar elements in {len(hostElements)} host "
            f"elements; bonded length {bondedLength:.6g}, unbonded length {unbondedLength:.6g}",
            name,
        )


def _splitBarAtHostBoundaries(bar, locator: HostElementLocator, model: FEModel) -> list:
    """Replace a bar element crossing host element boundaries by bar elements of the same type with nodes at the
    crossings, in the model and in all element sets containing it.

    Returns
    -------
    list
        The new bar elements, or the bar element itself if it lies in one host element.
    """
    barShape, coordinates = barGeometry(bar)
    parts, _ = splitBarElementAtHostBoundaries(bar, locator)

    # break points: the boundaries of the embedded parts, including those to unembedded stretches
    breakPoints = [-1.0]
    for eta in sorted([p.etaStart for p in parts] + [p.etaEnd for p in parts] + [1.0]):
        if eta - breakPoints[-1] > MIN_PART_LENGTH:
            breakPoints.append(eta)
    breakPoints[-1] = 1.0
    if len(breakPoints) == 2:
        return [bar]

    def newNode(eta):
        (label,) = model.topology.reserveNodeNumbers(1)
        node = Node(label, barPoint(barShape, coordinates, eta))
        model.createNode(node)
        if "all" in model.nodeSets:  # the fields are bundled for the nodes of 'all'
            model.nodeSets["all"].add(node)
        return node

    endNodes = [bar.nodes[0]] + [newNode(eta) for eta in breakPoints[1:-1]] + [bar.nodes[1]]

    newBars = []
    for i in range(len(breakPoints) - 1):
        nodes = [endNodes[i], endNodes[i + 1]]
        if barShape == "bar3":
            nodes.append(newNode(0.5 * (breakPoints[i] + breakPoints[i + 1])))
        (label,) = model.topology.reserveElementNumbers(1)
        newBar = type(bar)(bar.elType, label)
        newBar.setNodes(nodes)
        model.createElement(newBar)
        newBars.append(newBar)

    for elementSet in model.elementSets.values():
        if bar in elementSet:
            members = []
            for el in elementSet:
                members += newBars if el is bar else [el]
            elementSet.replaceMembers(members)
    model.removeElement(bar.elNumber)

    if barShape == "bar3":
        # the mid node of the original bar is not part of the new bars
        midNode = bar.nodes[2]
        if not any(midNode in el.nodes for el in model.elements.values()):
            del model.nodes[midNode.label]
            for nodeSet in model.nodeSets.values():
                if midNode in nodeSet:
                    nodeSet.replaceMembers([n for n in nodeSet if n is not midNode])

    return newBars


class _BarChain:
    """An unbranched chain of bar elements, ordered along the chain, with their orientation."""

    def __init__(self, elements: list):
        #: (bar element, reversed w.r.t. the chain direction)
        self.elements = elements
        self.shape = elements[0][0].ensightType
        self.lengths = []
        for bar, _ in elements:
            shape, coordinates = barGeometry(bar)
            if shape != self.shape:
                raise ValueError("A chain of bars must not mix linear and quadratic bar elements.")
            self.lengths.append(barLength(shape, coordinates))
        #: reference arc length at the start of each element, and at the chain end
        self.S = np.concatenate([[0.0], np.cumsum(self.lengths)])

    def windowNodes(self, k0: int, k1: int) -> list:
        """The unique nodes of the elements k0..k1 in chain order (bar2: e0 e1 ...; bar3: e0 m0 e1 m1 ...)."""
        nodes = []
        for bar, reversed_ in self.elements[k0 : k1 + 1]:
            start, end = (bar.nodes[1], bar.nodes[0]) if reversed_ else (bar.nodes[0], bar.nodes[1])
            if not nodes:
                nodes.append(start)
            if self.shape == "bar3":
                nodes.append(bar.nodes[2])
            nodes.append(end)
        return nodes


def _chainBars(bars: list) -> list:
    """Order bar elements into unbranched chains connected by their end nodes."""
    elementsAtNode = {}
    for bar in bars:
        for node in bar.nodes[:2]:
            elementsAtNode.setdefault(node, []).append(bar)
    for node, elements in elementsAtNode.items():
        if len(elements) > 2:
            raise ValueError(f"Large slip needs unbranched bars, but node {node.label} joins {len(elements)} bars.")

    remaining = {id(b): b for b in bars}
    chains = []
    while remaining:
        # start at a free end, if there is one
        start = next(
            (b for b in remaining.values() if any(len(elementsAtNode[n]) == 1 for n in b.nodes[:2])),
            next(iter(remaining.values())),
        )
        startNode = start.nodes[0] if len(elementsAtNode[start.nodes[0]]) == 1 else start.nodes[1]
        chain, bar, node = [], start, startNode
        while bar is not None and id(bar) in remaining:
            del remaining[id(bar)]
            reversed_ = bar.nodes[0] is not node
            chain.append((bar, reversed_))
            node = bar.nodes[0] if reversed_ else bar.nodes[1]
            bar = next((b for b in elementsAtNode[node] if b is not bar and id(b) in remaining), None)
        chains.append(_BarChain(chain))
    return chains
