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

The generator splits every bar (truss) element of ``embeddedElSet`` at the boundaries of the host elements of
``hostElSet`` and creates one Marmot ``EmbeddedBondElement`` per part. Its nodes are the nodes of the bar element
followed by the nodes of the host element; the bar keeps its own, independent displacement field, and the bond
element transfers the bond stress of a pluggable bond-slip law (e.g., ``LINEARELASTICBONDSLIP``,
``MODELCODE2010BONDSLIP``) between bar and host. The bar and host meshes are independent of each other.

Parts of a bar lying outside of all host elements (e.g., a free end of a pull-out specimen) remain unbonded.

The bars may also be beam elements (e.g., Marmot ``B2D2``, ``B3D3``, reported as ``bar2``/``bar3``): the bond elements couple
only their displacement field, the beam rotations are not coupled to the host (as for ``rotations=free`` of the
embedded region constraint), and the bond sees the linear interpolation of the beam axis between its nodes, not the
beam's cubic deflection -- keep ``splitBars=True`` (or a beam mesh finer than the host) for bent beams. Split beams
get the type and the section of the original beam.

With ``splitBars=True`` (the default), every bar element crossing host element boundaries is first replaced by bar
elements (of the same type) with nodes at the crossings, so that each bar element lies in exactly one host element,
and a stiff bond converges to the embedded region constraint (perfect bond) for the split bar mesh. Without splitting
(``splitBars=False``), a stiff bond forces the host displacement along a bar element to follow the bar element's
(linear) interpolation also between its nodes, which over-constrains (locks) the host for stiff bond laws, in
particular in the normal direction; use it only for bar meshes that already conform to the host mesh, or for soft
bond.

Host elements: ``tria3`` (CPS3, CPE3), ``quad4``, ``quad8``, ``tetra4`` (C3D4), ``hexa8`` and ``hexa20`` continuum
elements, i.e., the shapes of the Marmot continuum elements. Mesh refinement (``hAdaptivity``) of host elements is not supported and rejected.

Output: a bond element is drawn as its bar (the first nodes of a Marmot bond element are the nodes of the bar
element, followed by the host nodes, so the EnSight shape ``bar2``/``bar3`` takes the first 2 or 3 nodes). Without
splitting, several bond elements cover parts of the same bar element and are each drawn as the whole bar.

The generator needs the bar and host elements, so it must run after the manual mesh generation
(``executeAfterManualGeneration=True``). It creates the element set ``<name>_bond`` and a section assigning the
bond-slip law and the bond element properties (perimeter, part of the bar element, integration points).

For perfect bond, use the ``embeddedRegion`` constraint (:mod:`edelweissfe.constraints.embeddedregion`) instead, or a
stiff ``LINEARELASTICBONDSLIP`` law.

**Large slip.** With ``largeSlip=True`` (and ``maxSlip``), Marmot ``EmbeddedLargeSlipBondElement`` elements are
created instead, so that the bars may slide far, e.g., out of the concrete. The vocabulary is the one of the Marmot
element:

- *bar*: one unbranched polyline of bar elements (truss or beam, linear or quadratic, any orientation), found by
  chaining the bar elements at their end nodes;
- *channel*: the original path of a bar inside a host element, fixed in the host; it carries the bond history;
- *window*: the consecutive bar elements within ``maxSlip`` of a channel, among which the channel points search their
  *partner*, the bar point currently next to them;
- *fade*: the bond of a channel point fades smoothly (:math:`f = 3x^2 - 2x^3` of the covered fraction :math:`x` of its
  tributary part) as a bar end passes it, so the bonded length shrinks as the bar is pulled out.

One bond element couples a host element to the window of its channel
(``EBLS<nDim>D<nBarNodes><host>W<n>``). The window has about ``2 maxSlip / bar element length`` + 1 bar elements, at
most the largest window Marmot registers (48): choose ``maxSlip`` a little above the largest slip expected, and bar
elements not much shorter than ``maxSlip / 20``. A slip beyond the window makes Marmot throw; EdelweissFE then cuts the
increment back and stops at the minimum increment. The bars must be straight or smoothly curved (kinks above
``MAX_KINK_ANGLE`` and closed loops are rejected: at a kink the partner jumps, and a kinked bar cannot slide through a
kinked channel).

.. code-block:: edelweiss
    :caption: Example (large slip):

    *modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
        embeddedElSet=rebars
        hostElSet=concrete
        material=bond
        perimeter=50.27
        largeSlip=True
        ** the largest slip expected, here a complete pull-out of 30 mm; sets the window
        maxSlip=35.0

.. code-block:: edelweiss
    :caption: Example:

    ** units N, mm; Ø12 bar
    *material, name=ModelCode2010BondSlip, id=bond, provider=marmot
        ** tauMax, s1, s2, s3, alpha, tauF, K0, Kn: fck = 30 MPa, good bond, pull-out failure
        ** (fib MC2010 Tab. 6.1-1, s3 = clear rib spacing); K0, Kn are penalty choices
            13.69, 1.0, 2.0, 10.0, 0.4, 5.48, 200.0, 1000.0

    *modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
        embeddedElSet=rebars
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
from edelweissfe.sections.base.sectionbase import Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.embedding import (
    MIN_PART_LENGTH,
    HostElementLocator,
    barGeometry,
    barLength,
    barPoint,
    findBarPartsInHosts,
)
from edelweissfe.utils.isoparametricshapes import (
    INSIDE_TOLERANCE,
    shapeFunctionsAndDerivatives,
)
from edelweissfe.utils.schema import schemaField

#: Largest angle between consecutive bar elements of a bar with large slip (radians, 5 degrees).
MAX_KINK_ANGLE = np.radians(5.0)

#: The abbreviation of a host element shape in the Marmot bond element names.
HOST_SHAPE_CODES = {
    "tria3": "T3",
    "quad4": "Q4",
    "quad8": "Q8",
    "tetra4": "T4",
    "hexa8": "H8",
    "hexa20": "H20",
}


def largeSlipBondElementType(nDim: int, barShape: str, hostShape: str, window: int) -> str:
    """The name of the Marmot large-slip bond element, ``EBLS<nDim>D<number of bar nodes><host shape>W<window>``, e.g.,
    ``EBLS2D2Q4W3``."""
    return "EBLS" + bondElementType(nDim, barShape, hostShape)[len("EB") :] + f"W{window}"


def bondElementType(nDim: int, barShape: str, hostShape: str) -> str:
    """The name of the Marmot bond element, ``EB<nDim>D<number of bar nodes><host shape>``, e.g., ``EB2D2Q4``."""
    try:
        return f"EB{nDim}D{2 if barShape == 'bar2' else 3}{HOST_SHAPE_CODES[hostShape]}"
    except KeyError:
        raise NotImplementedError(f"No bond element for '{hostShape}' host elements.")


@dataclass(frozen=True)
class EmbeddedBondSchema:
    """The options this generator accepts, owned by this module and never mutated from outside it."""

    embeddedElSet: str | None = schemaField(
        description="The element set of the embedded bar (truss or beam) elements.",
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
        default=INSIDE_TOLERANCE,
    )
    largeSlip: bool = schemaField(
        description="Large slip: the bars may slide out of (or into) their hosts, see the module documentation. "
        "Requires maxSlip.",
        dtype=bool,
        default=False,
    )
    maxSlip: float | None = schemaField(
        description="Large slip only: the largest slip expected, which sets the window of bar elements each bond "
        "element couples to; a larger slip makes the analysis cut back and stop.",
        dtype=float,
        default=None,
    )
    splitBars: bool = schemaField(
        description="Replace the bar elements by bar elements with nodes at the crossings of the host element "
        "boundaries, such that every bar element lies in one host element. Without splitting, a stiff bond locks the "
        "host; switch it off only for bar meshes that conform to the host mesh, or for soft bond.",
        dtype=bool,
        default=True,
    )


class EmbeddedBondSection(Section):
    """Assigns the bond-slip law and the per-element properties to the generated bond elements.

    Unlike the sections of the input file, the element properties differ from element to element (the part of the
    bar element they cover), so this section is created by the generator itself and registered in
    ``model.sections``; it is applied with all other sections in :meth:`~edelweissfe.models.femodel.FEModel.prepareYourself`.

    Parameters
    ----------
    name
        The name of the section.
    model
        The model.
    material
        The Marmot bond-slip law, ``{"name": ..., "properties": ...}``.
    elementSet
        The bond elements.
    properties
        The element properties per bond element label.
    """

    #: The bond elements reference the nodes of their host elements, which a refinement of the hosts would replace.
    supportsHostRefinement = False

    def __init__(self, name: str, model, material: dict, elementSet: ElementSet, properties: dict):
        super().__init__(name, model, material, [elementSet])
        self.name = name
        self.properties = properties

    def assignSectionPropertiesToElement(self, element, material: dict):
        """Assign the per-element properties and the (possibly modified, e.g., by a ``changeMaterialProperty``
        step action) bond-slip law to a bond element: properties first, as the number of integration points sets
        the number of bond-slip law instances."""
        element.setProperties(self.properties[element.elNumber])
        element.initializeElement()
        self.assignMaterialToElement(element, material)


class Generator(GeneratorBase):
    """Creates bond-slip elements between embedded reinforcement bars and host continuum elements."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = EmbeddedBondSchema

    def __init__(
        self, name: str, model: FEModel, journal: Journal, *, configuration: EmbeddedBondSchema = EmbeddedBondSchema()
    ):
        if configuration.embeddedElSet not in model.elementSets or configuration.hostElSet not in model.elementSets:
            raise ValueError(
                f"{name}: element sets '{configuration.embeddedElSet}' and '{configuration.hostElSet}' must exist; "
                "run this generator with executeAfterManualGeneration=True."
            )
        material = model.materials[configuration.material]
        if not isinstance(material, dict):
            raise ValueError(f"{name}: the bond-slip law '{configuration.material}' must be a Marmot material.")

        if configuration.maxSlip is not None and not configuration.largeSlip:
            raise ValueError(f"{name}: maxSlip is an option of largeSlip=True.")
        if configuration.largeSlip and not (configuration.maxSlip is not None and configuration.maxSlip > 0):
            raise ValueError(f"{name}: largeSlip requires a positive maxSlip.")

        barElements = list(model.elementSets[configuration.embeddedElSet])
        hostElements = list(model.elementSets[configuration.hostElSet])
        locator = HostElementLocator(hostElements, tolerance=configuration.tolerance)
        nDim = model.domainSize

        if configuration.splitBars:
            barElements = [newBar for bar in barElements for newBar in remeshBarAtHostCrossings(bar, locator, model)]

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

        def findParts(barElement):
            """the parts of a bar element in the hosts, with the bonded and unbonded lengths accounted for"""
            nonlocal bondedLength, unbondedLength
            barShape, barCoordinates = barGeometry(barElement)
            parts, unbonded = findBarPartsInHosts(barElement, locator)
            bondedLength += sum(barLength(barShape, barCoordinates, p.etaStart, p.etaEnd) for p in parts)
            unbondedLength += unbonded
            return parts

        def createLargeSlipBondElement(bar, k: int, reversed_: bool, part):
            """the large-slip bond element of the channel of bar element k of the bar in part.hostElement"""
            # the channel in the parameter of the bar, c = k + (eta + 1) / 2 with eta along the bar
            etas = sorted((-part.etaEnd, -part.etaStart) if reversed_ else (part.etaStart, part.etaEnd))
            cStart, cEnd = (k + 0.5 * (eta + 1) for eta in etas)
            firstWindowElement, lastWindowElement = bar.window(cStart, cEnd, configuration.maxSlip)
            window = lastWindowElement - firstWindowElement + 1
            windowNodes = bar.nodes(firstWindowElement, lastWindowElement)
            host = part.hostElement
            checkIndependent(windowNodes, host)
            elType = largeSlipBondElementType(nDim, bar.shape, host.ensightType, window)
            try:
                createBondElement(
                    elType,
                    list(host.nodes) + windowNodes,
                    [
                        configuration.perimeter,
                        cStart - firstWindowElement,
                        cEnd - firstWindowElement,
                        float(firstWindowElement == 0),
                        float(lastWindowElement == len(bar.elements) - 1),
                    ],
                )
            except NotImplementedError:
                # the windows Marmot registers are the one source of the cap (maxWindow in Marmot)
                raise ValueError(
                    f"{name}: maxSlip={configuration.maxSlip} needs a window of {window} bar elements, but Marmot has "
                    f"no {elType}; reduce maxSlip or use longer bar elements."
                )

        if configuration.largeSlip:
            for bar in _unbranchedBars(barElements):
                for k, (barElement, reversed_) in enumerate(bar.elements):
                    for part in findParts(barElement):
                        createLargeSlipBondElement(bar, k, reversed_, part)
        else:
            for barElement in barElements:
                for part in findParts(barElement):
                    checkIndependent(barElement.nodes, part.hostElement)
                    createBondElement(
                        bondElementType(nDim, barElement.ensightType, part.hostElement.ensightType),
                        list(barElement.nodes) + list(part.hostElement.nodes),
                        [configuration.perimeter, part.etaStart, part.etaEnd],
                    )

        elementSet = ElementSet(f"{name}_bond", bondElements)
        model.elementSets[elementSet.name] = elementSet
        model.sections[f"{name}_section"] = EmbeddedBondSection(
            f"{name}_section", model, material, elementSet, properties
        )

        journal.message(
            f"{len(bondElements)} bond elements for {len(barElements)} bar elements in {len(hostElements)} host "
            f"elements; bonded length {bondedLength:.6g}, unbonded length {unbondedLength:.6g}",
            name,
        )


def remeshBarAtHostCrossings(bar, locator: HostElementLocator, model: FEModel) -> list:
    """Replace a bar element crossing host element boundaries by bar elements of the same type with nodes at the
    crossings, in the model and in all element sets containing it.

    Returns
    -------
    list
        The new bar elements, or the bar element itself if it lies in one host element.
    """
    barShape, coordinates = barGeometry(bar)
    parts, _ = findBarPartsInHosts(bar, locator)

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
        model.nodeSets["all"].add(node)  # the fields are bundled for the nodes of 'all' (exists after the mesh)
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


class _Bar:
    """One unbranched bar: its bar elements ordered along it, with their orientation.

    Parameters
    ----------
    elements
        (bar element, reversed w.r.t. the bar direction), ordered along the bar.
    """

    def __init__(self, elements: list):
        #: (bar element, reversed w.r.t. the bar direction)
        self.elements = elements
        self.shape = elements[0][0].ensightType
        self.lengths = []
        previousEndTangent = None
        for bar, reversed_ in elements:
            shape, coordinates = barGeometry(bar)
            if shape != self.shape:
                raise ValueError("A bar must not mix linear and quadratic bar elements.")
            self.lengths.append(barLength(shape, coordinates))
            # tangents at the start and the end of the element, along the bar
            tangents = [shapeFunctionsAndDerivatives(shape, eta)[1][:, 0] @ coordinates for eta in (-1.0, 1.0)]
            startTangent, endTangent = (-tangents[1], -tangents[0]) if reversed_ else tangents
            if previousEndTangent is not None:
                cosine = (
                    previousEndTangent
                    @ startTangent
                    / np.linalg.norm(previousEndTangent)
                    / np.linalg.norm(startTangent)
                )
                if np.arccos(np.clip(cosine, -1.0, 1.0)) > MAX_KINK_ANGLE:
                    raise ValueError(
                        f"Large slip needs straight or smoothly curved bars, but the bar has a kink of "
                        f"{np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))):.3g} degrees at element {bar.elNumber}."
                    )
            previousEndTangent = endTangent
        #: reference arc length at the start of each element, and at the end of the bar
        self.S = np.concatenate([[0.0], np.cumsum(self.lengths)])

    def window(self, cStart: float, cEnd: float, maxSlip: float) -> tuple[int, int]:
        """The first and the last bar element of the window of a channel: all bar elements within maxSlip of the channel
        [cStart, cEnd] (in the parameter c = k + (eta + 1) / 2 of the bar).

        The arc lengths of the channel ends are taken linear in c within the bar element, which is approximate for
        curved quadratic elements but only sizes the window."""
        k = min(int(np.floor(cStart)), len(self.elements) - 1)
        SStart = self.S[k] + (cStart - k) * self.lengths[k]
        SEnd = self.S[k] + (cEnd - k) * self.lengths[k]
        first, last = k, k
        while first > 0 and self.S[first] > SStart - maxSlip:
            first -= 1
        while last < len(self.elements) - 1 and self.S[last + 1] < SEnd + maxSlip:
            last += 1
        return first, last

    def nodes(self, first: int, last: int) -> list:
        """The unique nodes of the bar elements first..last along the bar (bar2: e0 e1 ...; bar3: e0 m0 e1 m1 ...)."""
        nodes = []
        for bar, reversed_ in self.elements[first : last + 1]:
            start, end = (bar.nodes[1], bar.nodes[0]) if reversed_ else (bar.nodes[0], bar.nodes[1])
            if not nodes:
                nodes.append(start)
            if self.shape == "bar3":
                nodes.append(bar.nodes[2])
            nodes.append(end)
        return nodes


def _unbranchedBars(barElements: list) -> list:
    """Order bar elements into unbranched bars connected by their end nodes.

    Raises
    ------
    ValueError
        For branches (a node joining more than two bar elements) and closed loops (no bar end to start from).
    """
    elementsAtNode = {}
    for bar in barElements:
        for node in bar.nodes[:2]:
            elementsAtNode.setdefault(node, []).append(bar)
    for node, elements in elementsAtNode.items():
        if len(elements) > 2:
            raise ValueError(f"Large slip needs unbranched bars, but node {node.label} joins {len(elements)} bars.")

    remaining = {id(b): b for b in barElements}
    bars = []
    while remaining:
        start = next((b for b in remaining.values() if any(len(elementsAtNode[n]) == 1 for n in b.nodes[:2])), None)
        if start is None:
            raise ValueError("Large slip needs bars with two ends, but the bar elements form a closed loop.")
        node = start.nodes[0] if len(elementsAtNode[start.nodes[0]]) == 1 else start.nodes[1]
        elements, bar = [], start
        while bar is not None:
            del remaining[id(bar)]
            reversed_ = bar.nodes[0] is not node
            elements.append((bar, reversed_))
            node = bar.nodes[0] if reversed_ else bar.nodes[1]
            bar = next((b for b in elementsAtNode[node] if b is not bar and id(b) in remaining), None)
        bars.append(_Bar(elements))
    return bars
