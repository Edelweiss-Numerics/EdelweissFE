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
A section for beam elements, e.g., the Marmot Euler-Bernoulli beams ``B2D2``, ``B2D3``, ``B3D2``, ``B3D3``,
``B2D2CR`` (co-rotational),
which integrate their cross section over section points with one material instance per point (any Marmot
hypoelastic material). This section generates the points (see :mod:`edelweissfe.utils.beamsections`) and passes
them, with the orientation and the torsion constant, as element properties.

Profiles (``profile=``):

* ``rectangle`` (``b`` along z, ``h`` along y; ``nY`` x ``nZ`` points),
* ``circle`` (``d``) and ``tube`` (``d``, ``di``; ``nRings`` x ``nSectors`` points),
* ``iprofile`` (``h``, ``b``, ``tf``, ``tw``; ``nY`` points along the web, ``nZ`` across a flange, ``nT`` through a
  flange or the web thickness),
* ``polygon`` (``polygonFile``: vertices ``y, z`` per line, loops separated by empty lines, holes by the even-odd
  rule; ``meshSize``, ``triangleOrder``),
* ``points`` (``pointsFile``: ``y, z, area`` per line, or ``y, area``),
* ``generic`` (``area``, ``Iz`` (2D: also ``I``), 3D: ``Iy``; 2 or 4 points reproducing them).

The 1D rule of the parametric profiles is ``rule=gauss|lobatto|simpson`` (lobatto and simpson place points on the
outer fibers, which detects the first yield exactly). The torsion constant ``J`` (3D) is computed for rectangles
(Roark), circles and tubes (exact) and I-profiles (thin-walled, :math:`\\sum b t^3 / 3`); it is required for the
other profiles. With ``centroidAtAxis=True`` (default) the points are shifted such that their centroid lies on the
beam axis (the line through the nodes); otherwise the given coordinates are measured from the axis (an eccentric
beam). The local y axis lies in the plane of a 2D beam; in 3D it is the part of the orientation vector ``(n1x, n1y,
n1z)`` normal to the beam axis, and :math:`z = x \\times y`. Without an orientation vector, :math:`(0, 0, 1)` is
used, or :math:`(0, 1, 0)` for beams (almost) parallel to the global z axis. In 2D, the z coordinates of the points
are dropped (use ``nZ=1`` for rectangles).

``multiplicity=k`` makes the beam stand for k identical parallel members (the point areas, A, I and J are scaled by
k, the coordinates are not), e.g., k fibres stacked through the thickness of a 2D model, as the area of a truss.

The co-rotational (finite rotation) variant ``B2D2CR`` takes the same section.

One material per element: a composite section (e.g., steel and concrete) is modeled by several beam elements on
the same nodes, one per material, each with its part of the section (the fiber integration is additive).

.. code-block:: edelweiss
    :caption: Example:

    *material, name=VonMises, id=steel, provider=marmot
        200000, 0.3, 500, 1000, 0, 1, 7.85e-9

    *section, name=pile, type=beam, profile=tube, d=300, di=260, nRings=3, nSectors=16, rule=lobatto, material=steel
        piles
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils import beamsections
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField

#: The profiles of the beam section.
PROFILES = ("rectangle", "circle", "tube", "iprofile", "polygon", "points", "generic")


def rectangleTorsionConstant(b: float, h: float) -> float:
    """The torsion constant of a solid rectangle (Roark's approximation, error below 0.5 %)."""
    a, c = max(b, h), min(b, h)
    return a * c**3 * (1.0 / 3.0 - 0.21 * c / a * (1.0 - c**4 / (12.0 * a**4)))


@dataclass(frozen=True)
class BeamSectionSchema:
    """The options this section accepts, owned by this module and never mutated from outside it."""

    profile: str = schemaField(description=f"The cross section profile, one of {PROFILES}.", dtype=str, default=None)
    rule: str = schemaField(
        description="The 1D rule of the section points of parametric profiles: gauss, lobatto or simpson.",
        dtype=str,
        default="gauss",
    )
    b: float | None = schemaField(description="rectangle, iprofile: width along z", dtype=float, default=None)
    h: float | None = schemaField(description="rectangle, iprofile: height along y", dtype=float, default=None)
    d: float | None = schemaField(description="circle, tube: (outer) diameter", dtype=float, default=None)
    di: float = schemaField(description="tube: inner diameter", dtype=float, default=0.0)
    tf: float | None = schemaField(description="iprofile: flange thickness", dtype=float, default=None)
    tw: float | None = schemaField(description="iprofile: web thickness", dtype=float, default=None)
    nY: int = schemaField(description="rectangle, iprofile (web): points along y", dtype=int, default=5)
    nZ: int = schemaField(
        description="rectangle, iprofile (flange width): points along z (default: 5 in 3D, 1 in 2D)",
        dtype=int,
        default=0,
    )
    nT: int = schemaField(description="iprofile: points through a flange or the web thickness", dtype=int, default=2)
    nRings: int = schemaField(description="circle, tube: points in the radial direction", dtype=int, default=3)
    nSectors: int = schemaField(
        description="circle, tube: points in the circumferential direction", dtype=int, default=8
    )
    polygonFile: str | None = schemaField(description="polygon: the file of the loops", dtype=str, default=None)
    meshSize: float | None = schemaField(description="polygon: the largest cell size", dtype=float, default=None)
    triangleOrder: int = schemaField(description="polygon: the degree of the triangle rule", dtype=int, default=2)
    pointsFile: str | None = schemaField(description="points: the file of the points", dtype=str, default=None)
    area: float | None = schemaField(description="generic: cross section area", dtype=float, default=None)
    I: float | None = schemaField(  # noqa: E741
        description="generic, 2D: second moment of area for bending in the plane (synonym of Iz)",
        dtype=float,
        default=None,
    )
    Iy: float | None = schemaField(description="generic, 3D: int z^2 dA", dtype=float, default=None)
    Iz: float | None = schemaField(description="generic: int y^2 dA", dtype=float, default=None)
    J: float | None = schemaField(
        description="3D: torsion constant (overrides the computed one)", dtype=float, default=None
    )
    multiplicity: float = schemaField(
        description="The number of identical parallel members this beam stands for (scales the point areas, and thus "
        "A, I and J, but not the coordinates; e.g., k fibres stacked through the thickness of a 2D model).",
        dtype=float,
        default=1.0,
    )
    centroidAtAxis: bool = schemaField(
        description="Shift the section points such that their centroid lies on the beam axis.", dtype=bool, default=True
    )
    n1x: float | None = schemaField(
        description="3D: orientation vector of the local y axis, x", dtype=float, default=None
    )
    n1y: float | None = schemaField(
        description="3D: orientation vector of the local y axis, y", dtype=float, default=None
    )
    n1z: float | None = schemaField(
        description="3D: orientation vector of the local y axis, z", dtype=float, default=None
    )
    materialParameterFromField: tuple[MaterialParameterFromFieldSchema, ...] = subKeywordField(
        description="use material properties given by an analytical field",
        schema=MaterialParameterFromFieldSchema,
    )
    writeMaterialPropertiesToFile: tuple[WriteMaterialPropertiesToFileSchema, ...] = subKeywordField(
        description="export material properties to file",
        schema=WriteMaterialPropertiesToFileSchema,
    )
    elementSets: str | None = datalineField(
        description="elementSets as comma separated list of element sets for this section", required=True
    )


def sectionPoints(c: BeamSectionSchema, nDim: int) -> tuple[beamsections.SectionPoints, float | None]:
    """The section points and the torsion constant (``None`` if unknown) of a beam section definition."""

    def need(*names):
        missing = [n for n in names if getattr(c, n) is None]
        if missing:
            raise ValueError(f"A beam section of profile '{c.profile}' requires {missing}.")

    profile = (c.profile or "").lower()
    nZ = c.nZ or (5 if nDim == 3 else 1)
    J = None
    if profile == "rectangle":
        need("b", "h")
        points = beamsections.rectangle(c.b, c.h, c.nY, nZ, c.rule)
        J = rectangleTorsionConstant(c.b, c.h)
    elif profile in ("circle", "tube"):
        need("d")
        di = c.di if profile == "tube" else 0.0
        points = beamsections.tube(c.d, di, c.nRings, c.nSectors, c.rule)
        J = np.pi * (c.d**4 - di**4) / 32
    elif profile == "iprofile":
        need("h", "b", "tf", "tw")
        points = beamsections.iProfile(c.h, c.b, c.tf, c.tw, c.nY, nZ, c.nT, c.rule)
        J = (2 * c.b * c.tf**3 + (c.h - 2 * c.tf) * c.tw**3) / 3
    elif profile == "polygon":
        need("polygonFile")
        points = beamsections.polygon(beamsections.readPolygon(c.polygonFile), c.meshSize, c.triangleOrder)
    elif profile == "points":
        need("pointsFile")
        points = beamsections.readPoints(c.pointsFile)
    elif profile == "generic":
        Iz = c.Iz if c.Iz is not None else c.I
        if c.area is None or Iz is None or (nDim == 3 and c.Iy is None):
            raise ValueError("A generic beam section requires area and Iz (or I), and Iy in 3D.")
        points = beamsections.genericPoints(c.area, c.Iy if nDim == 3 else 0.0, Iz)
    else:
        raise ValueError(f"Unknown beam profile '{c.profile}', use one of {PROFILES}.")

    if c.centroidAtAxis:
        points = points.centered()
    if c.J is not None:
        J = c.J
    if c.multiplicity != 1.0:
        if c.multiplicity <= 0:
            raise ValueError("The multiplicity of a beam section must be positive.")
        points = beamsections.SectionPoints.fromArrays(points.y, points.z, points.area * c.multiplicity)
        J = J * c.multiplicity if J is not None else None
    return points, J


class Section(SectionBase):
    """A section for beam elements, assigning the section points, the orientation and the torsion constant."""

    #: Option schema for this section, per OptionSchemaProvider.
    schema = BeamSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: BeamSectionSchema = BeamSectionSchema(),
    ):
        """Constructible standalone, with no parser involvement.

        Parameters
        ----------
        name
            The name of this section.
        model
            The model tree.
        material
            The (Marmot) material assigned to this section.
        elementSets
            The element sets this section is applied to.
        configuration
            The options this section accepts, see :class:`BeamSectionSchema`.
        """
        super().__init__(
            name,
            model,
            material,
            elementSets,
            configuration.materialParameterFromField,
            configuration.writeMaterialPropertiesToFile,
        )
        self.name = name
        self.nDim = model.domainSize
        self.points, self.J = sectionPoints(configuration, self.nDim)
        if self.nDim == 3 and (self.J is None or self.J <= 0):
            raise ValueError(f"Beam section {name}: a positive torsion constant J is required for this profile in 3D.")

        n1 = (configuration.n1x, configuration.n1y, configuration.n1z)
        if any(c is not None for c in n1):
            self.orientation = np.array([c or 0.0 for c in n1], dtype=float)
            if np.linalg.norm(self.orientation) == 0:
                raise ValueError(f"Beam section {name}: the orientation vector must not be zero.")
        else:
            self.orientation = None

    def elementProperties(self, element) -> np.ndarray:
        """The Marmot element properties of a beam element of this section."""
        p = self.points
        if self.nDim == 2:
            return np.concatenate([[len(p)], np.column_stack([p.y, p.area]).ravel()])

        n1 = self.orientation
        if n1 is None:
            coordinates = np.array([n.coordinates for n in element.nodes[:2]], dtype=float)
            axis = coordinates[1] - coordinates[0]
            axis /= np.linalg.norm(axis)
            n1 = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        return np.concatenate([n1, [self.J, len(p)], np.column_stack([p.y, p.z, p.area]).ravel()])

    def assignSectionPropertiesToElement(self, element, **kwargs):
        material = kwargs.get("material", self.material)

        if element.ensightType not in ("bar2", "bar3") or "rotation" not in element.fields[0]:
            raise Exception(f"Beam section is incompatible with element {element.elNumber} ({element.elType}).")

        element.setProperties(np.asarray(self.elementProperties(element), dtype=float))
        element.initializeElement()
        if not isinstance(material, dict):
            raise Exception(f"Beam section {self.name}: beam elements require a Marmot material.")
        element.setMaterial(material["name"], material["properties"])
