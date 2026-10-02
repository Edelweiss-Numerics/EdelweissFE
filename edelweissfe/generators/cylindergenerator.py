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
#  Paul Hofer paul.hofer@uibk.ac.at
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
A structured hex mesh generator for cylindrical geometries.

The cylinder is extruded along one of the global coordinate axes, selected
with ``axis`` (``x``, ``y`` or ``z``; ``y`` by default). The cross section
(perpendicular to that axis) is meshed as an O-grid: a square core block in
the center, surrounded by ``nR`` concentric rings of quads that map the
square's boundary radially outward onto the circle. This avoids triangles
and a singular node on the axis, so the resulting quad mesh can be extruded
into a fully hexahedral mesh. The core block's own subdivision is chosen
automatically so its elements are similarly sized to the ring elements:

.. code-block:: console

nSets, elSets, surface: 'name'_top, _bottom, _outer, _all are automatically
generated. _top/_bottom refer to the two end faces (perpendicular to
``axis``), _outer to the lateral (mantle) surface at r=radius.

Additional nSets: 'name'_centerTop/_centerBottom, the single node on the
cylinder's axis at each end face; and, for the two global axes spanning the
cross section, 'name'_centerLine<A>Top/_centerLine<A>Bottom, the nodes on
the diametral line parallel to axis <A> passing through the cylinder's axis,
on the top and bottom end faces only. <A> is spelled in upper case, so the
default ``axis=y`` yields _centerLineXTop/_centerLineXBottom and
_centerLineZTop/_centerLineZBottom.

Example
-------

Generate meshes on the fly using the following syntax:

.. code-block:: edelweiss

    *job, name=job, domain=3d, solver=NIST

    *modelGenerator, generator=cylinderGenerator, name=gen
        radius  =5.0
        lY      =10.0
        nR      =4
        nY      =8
        elType  =C3D8

Extruding along another axis only takes ``axis``; ``lY`` and ``nY`` keep
their names but are then measured along that axis, and ``x0``/``y0``/``z0``
still give the center of the bottom face:

.. code-block:: edelweiss

    *modelGenerator, generator=cylinderGenerator, name=gen
        axis    =z
        z0      =-10.0
        radius  =5.0
        lY      =10.0
        nR      =4
        nY      =8
        elType  =C3D8
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class CylinderGeneratorSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    ``elType`` is declared ``required=True`` explicitly, but is still given a ``default=None`` so
    the schema remains constructible for the constructor's default argument.
    """

    axis: str = schemaField(
        description=(
            "Global coordinate axis the cylinder is extruded along: 'x', 'y' or 'z'. The remaining two axes "
            "span the cross section."
        ),
        dtype=str,
        default="y",
    )

    x0: float = schemaField(
        description=(
            "Origin along the x axis: the bottom face of the cylinder if axis='x', the center of its cross "
            "section otherwise."
        ),
        dtype=float,
        default=0.0,
    )
    y0: float = schemaField(
        description=(
            "Origin along the y axis: the bottom face of the cylinder if axis='y', the center of its cross "
            "section otherwise."
        ),
        dtype=float,
        default=0.0,
    )
    z0: float = schemaField(
        description=(
            "Origin along the z axis: the bottom face of the cylinder if axis='z', the center of its cross "
            "section otherwise."
        ),
        dtype=float,
        default=0.0,
    )

    radius: float = schemaField(description="Radius of the cylinder.", dtype=float, default=1.0)
    lY: float = schemaField(
        description="Height of the cylinder, measured along ``axis`` (not necessarily the y axis).",
        dtype=float,
        default=1.0,
    )

    nR: int = schemaField(
        description=(
            "Number of elements along the radius, i.e., along a straight line from the center to the outer "
            "(lateral) surface. This is split automatically between the central square core block and the "
            "concentric rings surrounding it, keeping their element sizes similar."
        ),
        dtype=int,
        default=4,
    )
    nY: int = schemaField(
        description="Number of elements along the height, i.e., along ``axis`` (not necessarily the y axis).",
        dtype=int,
        default=4,
    )

    coreFraction: float = schemaField(
        description="Half-width of the central square core block, as a fraction of the radius.",
        dtype=float,
        default=0.7,
    )
    curvedBoundary: bool = schemaField(
        description=(
            "For quadratic elements only: place mid-side nodes on the outer surface exactly on the circle "
            "instead of at the straight-line midpoint."
        ),
        dtype=bool,
        default=True,
    )

    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


def _resolveAxis(axis):
    """Resolve the extrusion axis into global component indices.

    Returns the index of the extrusion axis, the indices of the two global axes spanning the
    cross section (in ascending order, so the O-grid's local (u, v) coordinates map onto them),
    and whether the in-plane quad winding has to be reversed.

    The O-grid quads are counterclockwise in the local (u, v) plane, so their normal points along
    ``e_u x e_v``. That coincides with the extrusion direction only when (u, v, axis) is a
    right-handed triple, which for the ascending in-plane pairing holds for x (y,z) and z (x,y),
    but not for y (x,z). In the latter case the winding is reversed so the element normal always
    points from one layer to the next.
    """
    inPlaneAxes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

    axis = axis.strip().lower()
    if axis not in inPlaneAxes:
        raise Exception("axis must be one of 'x', 'y', 'z', got '{:}'.".format(axis))

    iAxis = "xyz".index(axis)
    iU, iV = inPlaneAxes[axis]

    return iAxis, iU, iV, axis == "y"


def _generateOGridMesh(radius, nCore, nRing, coreFraction):
    """Generate the in-plane O-grid quad mesh: a core square block surrounded by radial rings.

    Returns node coordinates (n, 2), quad connectivity (m, 4) (counterclockwise), and the
    number of nodes/quads per ring (``nBoundary``). The last ``nBoundary`` rows of the returned
    quad array are always the outermost ring, i.e., the elements touching the outer surface.
    """
    a = coreFraction * radius

    xs = np.linspace(-a, a, nCore + 1)
    ys = np.linspace(-a, a, nCore + 1)

    nodes = []

    def addNode(x, y):
        nodes.append((x, y))
        return len(nodes) - 1

    coreId = np.empty((nCore + 1, nCore + 1), dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            coreId[i, j] = addNode(x, y)

    quads = []
    for i in range(nCore):
        for j in range(nCore):
            quads.append((coreId[i, j], coreId[i + 1, j], coreId[i + 1, j + 1], coreId[i, j + 1]))

    # boundary loop of the core square, counterclockwise, no repeated nodes
    boundaryIJ = []
    boundaryIJ += [(i, 0) for i in range(0, nCore)]
    boundaryIJ += [(nCore, j) for j in range(0, nCore)]
    boundaryIJ += [(i, nCore) for i in range(nCore, 0, -1)]
    boundaryIJ += [(0, j) for j in range(nCore, 0, -1)]

    boundaryIds = [coreId[i, j] for i, j in boundaryIJ]
    boundaryXY = [(xs[i], ys[j]) for i, j in boundaryIJ]

    thetas = [math.atan2(y, x) for x, y in boundaryXY]
    d0s = [math.hypot(x, y) for x, y in boundaryXY]

    nBoundary = len(boundaryIds)
    prevIds = boundaryIds
    for k in range(1, nRing + 1):
        ringIds = []
        for theta, d0 in zip(thetas, d0s):
            r = d0 + k / nRing * (radius - d0)
            ringIds.append(addNode(r * math.cos(theta), r * math.sin(theta)))

        for idx in range(nBoundary):
            nxt = (idx + 1) % nBoundary
            quads.append((prevIds[idx], ringIds[idx], ringIds[nxt], prevIds[nxt]))

        prevIds = ringIds

    return np.array(nodes), np.array(quads, dtype=int), nBoundary


def _smoothOGridMesh(nodes, quads, radius, nIter=30, relax=0.5):
    """Laplacian-smooth the O-grid mesh to improve element quality.

    Interior nodes (including the former square-core boundary) are relaxed towards the average
    of their neighbors. Nodes on the outer surface are kept exactly on the circle, and are
    additionally allowed to slide tangentially to equalize angular spacing.
    """
    nodes = nodes.copy()
    nNodes = len(nodes)

    rows, cols = [], []
    for quad in quads:
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            rows += [a, b]
            cols += [b, a]
    adjacency = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nNodes, nNodes)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    averaging = sp.diags(1.0 / degree) @ adjacency

    radii = np.linalg.norm(nodes, axis=1)
    isBoundary = np.isclose(radii, radius, atol=radius * 1e-6)
    interior = ~isBoundary

    boundaryIdx = np.where(isBoundary)[0]
    order = boundaryIdx[np.argsort(np.arctan2(nodes[boundaryIdx, 1], nodes[boundaryIdx, 0]))]
    angles = np.arctan2(nodes[order, 1], nodes[order, 0])
    angles = np.unwrap(np.concatenate([angles, angles[:1] + 2 * np.pi]))[:-1]

    for _ in range(nIter):
        averaged = averaging @ nodes
        nodes[interior] = (1 - relax) * nodes[interior] + relax * averaged[interior]

        prevAngles = np.roll(angles, 1)
        prevAngles[0] -= 2 * np.pi
        nextAngles = np.roll(angles, -1)
        nextAngles[-1] += 2 * np.pi
        angles = (1 - relax) * angles + relax * 0.5 * (prevAngles + nextAngles)
        nodes[order] = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    return nodes


def _addMidsideNodes(nodes, quads, radius, curvedBoundary):
    """Elevate the 4-node quad mesh to 8-node (serendipity) quadratic elements.

    The original corner nodes keep their indices (0..n-1); mid-side nodes are appended
    afterwards, one per edge (shared edges get a single, shared mid-side node). Quad columns:
    4 corners (CCW) followed by the mid-side nodes of edges (0-1, 1-2, 2-3, 3-0).
    """
    nodesOut = [tuple(p) for p in nodes]
    radii = np.linalg.norm(nodes, axis=1)
    isBoundary = np.isclose(radii, radius, atol=radius * 1e-6)

    edgeMidnode = {}

    def getMidnode(a, b):
        key = (a, b) if a < b else (b, a)
        if key in edgeMidnode:
            return edgeMidnode[key]
        pa, pb = np.array(nodesOut[a]), np.array(nodesOut[b])
        if curvedBoundary and isBoundary[a] and isBoundary[b]:
            direction = pa / np.linalg.norm(pa) + pb / np.linalg.norm(pb)
            mid = radius * direction / np.linalg.norm(direction)
        else:
            mid = 0.5 * (pa + pb)
        idx = len(nodesOut)
        nodesOut.append(tuple(mid))
        edgeMidnode[key] = idx
        return idx

    quads8 = []
    for n0, n1, n2, n3 in quads:
        quads8.append((n0, n1, n2, n3, getMidnode(n0, n1), getMidnode(n1, n2), getMidnode(n2, n3), getMidnode(n3, n0)))

    return np.array(nodesOut), np.array(quads8, dtype=int)


class Generator(GeneratorBase):
    """A structured hex mesh generator for cylindrical geometries."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = CylinderGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: CylinderGeneratorSchema = CylinderGeneratorSchema(),
    ):
        """Constructible standalone, with no parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            The name of this generator instance, used as the prefix for the generated sets.
        model
            The model tree to populate. Mutated in place.
        journal
            Unused.
        configuration
            The options this generator accepts; ``elType`` is still required, see
            :class:`CylinderGeneratorSchema`.
        """
        iAxis, iU, iV, reverseWinding = _resolveAxis(configuration.axis)

        # the origin's axial component locates the bottom face, the two in-plane ones the center
        # of the cross section
        origin = np.array([configuration.x0, configuration.y0, configuration.z0])
        axisNames = "XYZ"

        radius = configuration.radius
        lY = configuration.lY

        nR = configuration.nR
        nY = configuration.nY

        coreFraction = configuration.coreFraction
        curvedBoundary = configuration.curvedBoundary

        if radius <= 0 or lY <= 0:
            raise Exception("radius and lY must be positive.")
        if nR < 2 or nY < 1:
            raise Exception("nR must be >= 2 (at least one element in the core block and one ring) and nY >= 1.")
        if not (0 < coreFraction < 1 / math.sqrt(2)):
            raise Exception("coreFraction must be in (0, 1/sqrt(2)) so the core block stays inside the cylinder.")

        # split nR (elements along the radius, center to outer surface) between the core block's own
        # half-width subdivision and the rings, keeping elements in both regions similarly sized: since
        # the core spans [0, coreFraction*radius] and the rings span [coreFraction*radius, radius], an
        # even element size along the full radius gives nCoreHalf = round(coreFraction * nR).
        nCoreHalf = max(1, min(nR - 1, round(coreFraction * nR)))
        nRing = nR - nCoreHalf
        nCore = 2 * nCoreHalf

        elTypeName = configuration.elType
        elProvider = configuration.elProvider
        elType = getElementClass(elTypeName, elProvider)

        testEl = elType(elTypeName, 0)

        if testEl.nNodes == 8:
            order = 1
        elif testEl.nNodes == 20:
            order = 2
        else:
            return

        # --- in-plane O-grid mesh (core block + radial rings) ---------------------
        nodesLin, quadsLin, nBoundary = _generateOGridMesh(radius, nCore, nRing, coreFraction)
        nodesLin = _smoothOGridMesh(nodesLin, quadsLin, radius)

        # quads are appended core-first, then ring by ring; the last `nBoundary` quads are
        # therefore always the outermost ring, i.e., the elements touching the outer surface.
        outerQuadMask = np.zeros(len(quadsLin), dtype=bool)
        outerQuadMask[-nBoundary:] = True

        if order == 1:
            nodes2D = nodesLin
            quads2D = quadsLin
        else:
            nodes2D, quads2D = _addMidsideNodes(nodesLin, quadsLin, radius, curvedBoundary)

        nCorners2D = len(nodesLin)

        # the last `nBoundary` corner nodes are, by construction, exactly the outer boundary loop.
        # For quadratic elements, the outer mid-side node of an outer-ring quad (c0,c1,c2,c3,m01,m12,m23,m30)
        # is m12 (the mid-node of edge c1-c2, which is always the outer edge of that quad, see the ring
        # construction in `_generateOGridMesh`). Determining the outer node set this way (rather than by
        # checking coordinates against `radius`) keeps it correct even if curvedBoundary=False.
        isOuterCorner2D = np.zeros(nCorners2D, dtype=bool)
        isOuterCorner2D[nCorners2D - nBoundary :] = True

        isOuter2D = np.zeros(len(nodes2D), dtype=bool)
        isOuter2D[:nCorners2D] = isOuterCorner2D
        if order == 2:
            isOuter2D[quads2D[outerQuadMask, 5]] = True

        # the mesh is exactly symmetric about both local axes (`nCore` is always even), so the two
        # diametral lines through the center, parallel to the global in-plane axes `iU` and `iV`,
        # consist of nodes with local v=0 and local u=0, respectively; their intersection is the
        # single center node. Only needed on the top/bottom (always "full") layers, hence based on
        # `nodes2D` only.
        tol = radius * 1e-6
        isCenterLineU2D = np.isclose(nodes2D[:, 1], 0.0, atol=tol)
        isCenterLineV2D = np.isclose(nodes2D[:, 0], 0.0, atol=tol)
        isCenter2D = isCenterLineU2D & isCenterLineV2D

        currentNodeLabel = 1
        if model.nodes:
            currentNodeLabel += max(model.nodes.keys())
        currentElementLabel = 1
        if model.elements:
            currentElementLabel += max(model.elements.keys())

        elements = []
        elementsTop = []
        elementsBottom = []
        elementsOuter = []
        nodesTop = []
        nodesBottom = []
        nodesOuter = []
        nodesCenterBottom = []
        nodesCenterTop = []
        nodesCenterLineUTop = []
        nodesCenterLineUBottom = []
        nodesCenterLineVTop = []
        nodesCenterLineVBottom = []

        def makeCoordinates(u, v, axial):
            """Scatter the local (in-plane, axial) coordinates onto the global axes."""
            coordinates = np.empty(3)
            coordinates[iU] = origin[iU] + u
            coordinates[iV] = origin[iV] + v
            coordinates[iAxis] = axial
            return coordinates

        def windCorners(c0, c1, c2, c3):
            """Order the four in-plane corners of a quad for the extruded hex element.

            Two things have to hold at once. The element normal must point from one layer to the
            next, which for the reversed case means flipping the in-plane order. And the quad's
            outer edge -- always c1-c2, see the ring construction in `_generateOGridMesh` -- must
            end up as the local edge 2-3, so that the outer surface is face S5 for either winding;
            the non-reversed case therefore rotates rather than leaving the order untouched, which
            preserves the orientation.
            """
            return (c0, c3, c2, c1) if reverseWinding else (c3, c0, c1, c2)

        def windMidnodes(m01, m12, m23, m30):
            """The mid-side nodes of the edges of :func:`windCorners`'s output, in the same order."""
            return (m30, m23, m12, m01) if reverseWinding else (m30, m01, m12, m23)

        if order == 1:
            nNodesY = nY + 1
            yLayers = np.linspace(origin[iAxis], origin[iAxis] + lY, nNodesY)

            layerNodes = []
            for iy in range(nNodesY):
                layer = []
                for U, V in nodes2D:
                    node = Node(currentNodeLabel, makeCoordinates(U, V, yLayers[iy]))
                    layer.append(node)
                    model.nodes[currentNodeLabel] = node
                    currentNodeLabel += 1
                layerNodes.append(layer)
                if iy == 0:
                    nodesBottom.extend(layer)
                    nodesCenterBottom.extend(n for n, isC in zip(layer, isCenter2D) if isC)
                    nodesCenterLineUBottom.extend(n for n, isC in zip(layer, isCenterLineU2D) if isC)
                    nodesCenterLineVBottom.extend(n for n, isC in zip(layer, isCenterLineV2D) if isC)
                if iy == nNodesY - 1:
                    nodesTop.extend(layer)
                    nodesCenterTop.extend(n for n, isC in zip(layer, isCenter2D) if isC)
                    nodesCenterLineUTop.extend(n for n, isC in zip(layer, isCenterLineU2D) if isC)
                    nodesCenterLineVTop.extend(n for n, isC in zip(layer, isCenterLineV2D) if isC)
                nodesOuter.extend(n for n, isOuter in zip(layer, isOuter2D) if isOuter)

            for iy in range(nY):
                for iq, (c0, c1, c2, c3) in enumerate(quads2D):
                    rc = windCorners(c0, c1, c2, c3)
                    nodeList = [layerNodes[iy][idx] for idx in rc] + [layerNodes[iy + 1][idx] for idx in rc]

                    newEl = elType(elTypeName, currentElementLabel)
                    newEl.setNodes(nodeList)

                    elements.append(newEl)
                    model.elements[currentElementLabel] = newEl

                    if iy == 0:
                        elementsBottom.append(newEl)
                    if iy == nY - 1:
                        elementsTop.append(newEl)
                    if outerQuadMask[iq]:
                        elementsOuter.append(newEl)

                    currentElementLabel += 1

        else:
            nNodesYTotal = 2 * nY + 1
            yLayers = np.linspace(origin[iAxis], origin[iAxis] + lY, nNodesYTotal)

            layerNodes = []
            for t in range(nNodesYTotal):
                fullLayer = t % 2 == 0
                coordsUV = nodes2D if fullLayer else nodesLin
                isOuter = isOuter2D if fullLayer else isOuterCorner2D
                layer = []
                for U, V in coordsUV:
                    node = Node(currentNodeLabel, makeCoordinates(U, V, yLayers[t]))
                    layer.append(node)
                    model.nodes[currentNodeLabel] = node
                    currentNodeLabel += 1
                layerNodes.append(layer)
                if fullLayer:
                    if t == 0:
                        nodesBottom.extend(layer)
                        nodesCenterBottom.extend(n for n, isC in zip(layer, isCenter2D) if isC)
                        nodesCenterLineUBottom.extend(n for n, isC in zip(layer, isCenterLineU2D) if isC)
                        nodesCenterLineVBottom.extend(n for n, isC in zip(layer, isCenterLineV2D) if isC)
                    if t == nNodesYTotal - 1:
                        nodesTop.extend(layer)
                        nodesCenterTop.extend(n for n, isC in zip(layer, isCenter2D) if isC)
                        nodesCenterLineUTop.extend(n for n, isC in zip(layer, isCenterLineU2D) if isC)
                        nodesCenterLineVTop.extend(n for n, isC in zip(layer, isCenterLineV2D) if isC)
                nodesOuter.extend(n for n, isOut in zip(layer, isOuter) if isOut)

            for iy in range(nY):
                tBottom, tMid, tTop = 2 * iy, 2 * iy + 1, 2 * iy + 2
                for iq, (c0, c1, c2, c3, m01, m12, m23, m30) in enumerate(quads2D):
                    rc = windCorners(c0, c1, c2, c3)
                    # mid-side nodes of edges (rc0-rc1, rc1-rc2, rc2-rc3, rc3-rc0)
                    rm = windMidnodes(m01, m12, m23, m30)

                    bottomCorners = [layerNodes[tBottom][idx] for idx in rc]
                    topCorners = [layerNodes[tTop][idx] for idx in rc]
                    bottomMids = [layerNodes[tBottom][idx] for idx in rm]
                    topMids = [layerNodes[tTop][idx] for idx in rm]
                    verticalMids = [layerNodes[tMid][idx] for idx in rc]

                    nodeList = bottomCorners + topCorners + bottomMids + topMids + verticalMids

                    newEl = elType(elTypeName, currentElementLabel)
                    newEl.setNodes(nodeList)

                    elements.append(newEl)
                    model.elements[currentElementLabel] = newEl

                    if iy == 0:
                        elementsBottom.append(newEl)
                    if iy == nY - 1:
                        elementsTop.append(newEl)
                    if outerQuadMask[iq]:
                        elementsOuter.append(newEl)

                    currentElementLabel += 1

        model._populateNodeFieldVariablesFromElements()

        # node sets
        model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), nodesTop)
        model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), nodesBottom)
        model.nodeSets["{:}_outer".format(name)] = NodeSet("{:}_outer".format(name), nodesOuter)
        model.nodeSets["{:}_centerTop".format(name)] = NodeSet("{:}_centerTop".format(name), nodesCenterTop)
        model.nodeSets["{:}_centerBottom".format(name)] = NodeSet("{:}_centerBottom".format(name), nodesCenterBottom)
        # the two diametral center lines are named after the global axis they are parallel to, so
        # the default axis='y' keeps yielding _centerLineX* and _centerLineZ*
        for axisIdx, nodesTopLine, nodesBottomLine in (
            (iU, nodesCenterLineUTop, nodesCenterLineUBottom),
            (iV, nodesCenterLineVTop, nodesCenterLineVBottom),
        ):
            setName = "{:}_centerLine{:}Top".format(name, axisNames[axisIdx])
            model.nodeSets[setName] = NodeSet(setName, nodesTopLine)
            setName = "{:}_centerLine{:}Bottom".format(name, axisNames[axisIdx])
            model.nodeSets[setName] = NodeSet(setName, nodesBottomLine)

        # element sets
        model.elementSets["{:}_all".format(name)] = ElementSet("{:}_all".format(name), elements)
        model.elementSets["{:}_top".format(name)] = ElementSet("{:}_top".format(name), elementsTop)
        model.elementSets["{:}_bottom".format(name)] = ElementSet("{:}_bottom".format(name), elementsBottom)
        model.elementSets["{:}_outer".format(name)] = ElementSet("{:}_outer".format(name), elementsOuter)

        # surfaces: S1/S2 are the bottom/top faces, S5 the outward-radial face of the outer-ring elements
        # (both hold regardless of element order, since Abaqus face numbering only depends on corner connectivity)
        surfaceName = "{:}_bottom".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {1: model.elementSets[surfaceName]})
        surfaceName = "{:}_top".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {2: model.elementSets[surfaceName]})
        surfaceName = "{:}_outer".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {5: model.elementSets[surfaceName]})
