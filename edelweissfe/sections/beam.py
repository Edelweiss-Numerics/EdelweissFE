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
A section for beam elements: the Marmot Euler-Bernoulli beams ``B2D2``, ``B2D3``, ``B3D2``, ``B3D3`` and their
co-rotational (finite rotation) variants ``B2D2CR`` and ``B3D2CR``. They integrate their cross section over section
points (fibers), each with its own material instance (any Marmot hypoelastic material). This section generates the
fibers (see :mod:`edelweissfe.utils.beamsections`) and passes them, with the orientation and the torsion constant,
as element properties. The element documentation is Marmot's beam element page.

Profiles (``profile=``):

* ``rectangle`` (``b`` along z, ``h`` along y; ``nY`` x ``nZ`` fibers),
* ``circle`` (``d``) and ``tube`` (``d``, ``di``; ``nRings`` x ``nSectors`` fibers),
* ``iprofile`` (``h``, ``b``, ``tf``, ``tw``; ``nY`` fibers along the web, ``nZ`` across a flange, ``nT`` through a
  flange or the web thickness),
* ``polygon`` (``polygonFile``: vertices ``y, z`` per line, loops separated by empty lines, holes by the even-odd
  rule; ``meshSize``, ``triangleOrder``),
* ``points`` (``pointsFile``: ``y, z, area`` per line, or ``y, area``),
* ``generic`` (``area``, ``Iz``, 3D: ``Iy``; 2 or 4 fibers reproducing them).

The 1D rule of the parametric profiles is ``rule=gauss|lobatto|simpson`` (lobatto and simpson place fibers on the
outer edges, which detects the first yield exactly). The torsion constant ``J`` (3D) is computed for rectangles
(Roark), circles and tubes (exact) and I-profiles (thin-walled, :math:`\\sum b t^3 / 3`); it is required for the
other profiles. With ``centroidAtAxis=True`` (default) the fibers are shifted such that their centroid lies on the
beam axis (the line through the nodes); otherwise the given coordinates are measured from the axis (an eccentric
beam).

Local axes: the local y axis lies in the plane of a 2D beam; in 3D it is the part of the orientation vector ``(vx,
vy, vz)`` (Marmot's :math:`\\mathbf{v}`) normal to the beam axis, and :math:`z = x \\times y`. In 3D the
orientation vector is required, except for circles and tubes. In 2D, the z coordinates of the fibers are dropped
(use ``nZ=1`` for rectangles).

``multiplicity=k`` makes the beam stand for k identical parallel members (the fiber areas, A, I and J are scaled by
k, the coordinates are not), e.g., k fibers stacked through the thickness of a 2D model, as the area of a truss.

One material per element: a composite section (e.g., steel and concrete) is modeled by several beam elements on
the same nodes, one per material, each with its part of the section (the fiber integration is additive). The parts
must measure their coordinates from one common axis: give them ``centroidAtAxis=False`` (enforced). The torsion
constants of the parts are added, which is not the torsion constant of a composite (in particular a closed)
section; give ``J`` per part accordingly. An eccentric section (a non-concentric composite, or ``centroidAtAxis=False``)
couples axial strain and bending: the 3-node beams represent it exactly for a tip load, the 2-node beams (axial
strain constant per element) converge with :math:`h^2`.

The co-rotational variants take the same section. For ``B3D2CR`` the rotation dofs are total rotation vectors: a
nodal moment load on the ``rotation`` field acts on these dofs and is a physical moment only for small rotations
(apply finite-rotation moments as force couples).

Units: none of their own, use one consistent system, e.g., N, mm, MPa.

.. code-block:: edelweiss
    :caption: Example: a 2D cantilever, length 1000 mm, rectangle 100 x 200 mm, tip force 1 kN
        (testfiles/marmot/BeamCantilever2D; tip deflection P L^3 / (3 E I))

    *material, name=linearelastic, id=steel, provider=marmot
    ** E [MPa], nu
    200000, 0.3

    *node
    1, 0, 0
    2, 1000, 0

    *element, type=B2D2
    1, 1, 2

    *elSet, elSet=beam
    1

    *section, name=s, type=beam, profile=rectangle, b=100, h=200, nY=6, material=steel
    beam

    *step, solver=theSolver
    >>dirichlet, name=clamp, nSet=root, field=displacement, 1=0, 2=0
    >>dirichlet, name=clampR, nSet=root, field=rotation, 1=0
    >>nodeforces, name=P, nSet=tip, field=displacement, 2=-1000

Results of the quadrature points: ``normal force``, ``bending moment`` (3D: ``bending moments``, ``torque``),
``axial strain``, ``curvature``, ``section forces``, ``fiber k stress`` (:math:`\\sigma_{11}, \\sigma_{12},
\\sigma_{13}` of fiber k), ``elastic energy``, ``dissipation``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils import beamsections
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField


def rectangleTorsionConstant(b: float, h: float) -> float:
    """The torsion constant of a solid rectangle (Roark, *Formulas for Stress and Strain*, torsion of a solid
    rectangular section; error below 0.5 %)."""
    a, c = max(b, h), min(b, h)
    return a * c**3 * (1.0 / 3.0 - 0.21 * c / a * (1.0 - c**4 / (12.0 * a**4)))


@dataclass(frozen=True)
class _Profile:
    """A parametric profile of the beam section: everything about one shape in one place."""

    #: The options the profile requires, with their values: ``(configuration, nDim) -> {name: value}``.
    requiredOptions: Callable
    #: The section points: ``(configuration, nZ) -> SectionPoints``.
    points: Callable
    #: The torsion constant, or ``None`` if it must be given (``J``): ``configuration -> float``.
    torsionConstant: Callable | None = None
    #: Rotationally symmetric profiles need no orientation vector in 3D.
    rotationallySymmetric: bool = False


#: The profiles of the beam section. Adding a profile: a generator of its section points in
#: :mod:`edelweissfe.utils.beamsections`, its options in :class:`BeamSectionSchema`, an entry here (required options,
#: points, torsion constant with its reference) and a line in the module docstring.
PROFILE_TABLE = {
    "rectangle": _Profile(
        lambda c, nDim: {"b": c.b, "h": c.h},
        lambda c, nZ: beamsections.rectangle(c.b, c.h, c.nY, nZ, c.rule),
        lambda c: rectangleTorsionConstant(c.b, c.h),
    ),
    "circle": _Profile(
        lambda c, nDim: {"d": c.d},
        lambda c, nZ: beamsections.tube(c.d, 0.0, c.nRings, c.nSectors, c.rule),
        lambda c: np.pi * c.d**4 / 32,  # exact (polar moment)
        rotationallySymmetric=True,
    ),
    "tube": _Profile(
        lambda c, nDim: {"d": c.d},
        lambda c, nZ: beamsections.tube(c.d, c.di, c.nRings, c.nSectors, c.rule),
        lambda c: np.pi * (c.d**4 - c.di**4) / 32,  # exact (polar moment)
        rotationallySymmetric=True,
    ),
    "iprofile": _Profile(
        lambda c, nDim: {"h": c.h, "b": c.b, "tf": c.tf, "tw": c.tw},
        lambda c, nZ: beamsections.iProfile(c.h, c.b, c.tf, c.tw, c.nY, nZ, c.nT, c.rule),
        lambda c: (2 * c.b * c.tf**3 + (c.h - 2 * c.tf) * c.tw**3) / 3,  # thin-walled open section, sum b t^3 / 3
    ),
    "polygon": _Profile(
        lambda c, nDim: {"polygonFile": c.polygonFile},
        lambda c, nZ: beamsections.polygon(beamsections.readPolygon(c.polygonFile), c.meshSize, c.triangleOrder),
    ),
    "points": _Profile(
        lambda c, nDim: {"pointsFile": c.pointsFile},
        lambda c, nZ: beamsections.readPoints(c.pointsFile),
    ),
    "generic": _Profile(
        lambda c, nDim: {"area": c.area, "Iz": c.Iz} | ({"Iy": c.Iy} if nDim == 3 else {}),
        lambda c, nZ: beamsections.genericPoints(c.area, c.Iy if c.Iy is not None else 0.0, c.Iz),
    ),
}

#: The names of the profiles.
PROFILES = tuple(PROFILE_TABLE)


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
    Iy: float | None = schemaField(description="generic, 3D: int z^2 dA", dtype=float, default=None)
    Iz: float | None = schemaField(description="generic: int y^2 dA", dtype=float, default=None)
    J: float | None = schemaField(
        description="3D: torsion constant (overrides the computed one)", dtype=float, default=None
    )
    multiplicity: float = schemaField(
        description="The number of identical parallel members this beam stands for (scales the point areas, and thus "
        "A, I and J, but not the coordinates; e.g., k fibers stacked through the thickness of a 2D model).",
        dtype=float,
        default=1.0,
    )
    centroidAtAxis: bool = schemaField(
        description="Shift the section points such that their centroid lies on the beam axis.", dtype=bool, default=True
    )
    vx: float | None = schemaField(
        description="3D: orientation vector of the local y axis, x", dtype=float, default=None
    )
    vy: float | None = schemaField(
        description="3D: orientation vector of the local y axis, y", dtype=float, default=None
    )
    vz: float | None = schemaField(
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
    profileName = (c.profile or "").lower()
    if profileName not in PROFILE_TABLE:
        raise ValueError(f"Unknown beam profile '{c.profile}', use one of {PROFILES}.")
    profile = PROFILE_TABLE[profileName]
    missing = [name for name, value in profile.requiredOptions(c, nDim).items() if value is None]
    if missing:
        raise ValueError(f"A beam section of profile '{profileName}' requires {missing}.")

    nZ = c.nZ or (5 if nDim == 3 else 1)
    points = profile.points(c, nZ)
    J = c.J if c.J is not None else (profile.torsionConstant(c) if profile.torsionConstant else None)

    if c.centroidAtAxis:
        points = points.centered()
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

        v = (configuration.vx, configuration.vy, configuration.vz)
        if any(c is not None for c in v):
            self.orientation = np.array([c or 0.0 for c in v], dtype=float)
            if np.linalg.norm(self.orientation) == 0:
                raise ValueError(f"Beam section {name}: the orientation vector must not be zero.")
        else:
            self.orientation = None
            # a default orientation chosen per element jumps by 90 degrees along a curved member, which silently
            # rotates a non-circular section; only rotationally symmetric profiles may omit it
            if self.nDim == 3 and not PROFILE_TABLE[configuration.profile.lower()].rotationallySymmetric:
                raise ValueError(
                    f"Beam section {name}: a 3D section of profile '{configuration.profile}' requires the orientation "
                    "vector vx, vy, vz (the local y axis, Marmot's v)."
                )

        self.centroidAtAxis = configuration.centroidAtAxis
        self._checkCompositeParts(name, model, elementSets)

    def _checkCompositeParts(self, name, model, elementSets):
        """The parts of a composite section (beam elements on the same nodes) must measure their coordinates from
        one common axis: with ``centroidAtAxis=True`` each part would be shifted to its own centroid, and a
        non-concentric composite (e.g., a slab with a rebar layer) would silently become concentric parts."""
        self._elementNodes = {
            frozenset(node.label for node in element.nodes) for elementSet in elementSets for element in elementSet
        }
        for other in model.sections.values():
            if other is self or not isinstance(other, Section):
                continue
            if (self.centroidAtAxis or other.centroidAtAxis) and self._elementNodes & other._elementNodes:
                raise ValueError(
                    f"Beam sections {other.name} and {name} are parts of a composite section (elements on the same "
                    "nodes): give both centroidAtAxis=False and their section coordinates from the common beam axis."
                )

    def elementProperties(self, element) -> np.ndarray:
        """The Marmot element properties of a beam element of this section."""
        p = self.points
        if self.nDim == 2:
            return np.concatenate([[len(p)], np.column_stack([p.y, p.area]).ravel()])

        v = self.orientation
        if v is None:  # a rotationally symmetric section: any vector normal to the axis
            coordinates = np.array([n.coordinates for n in element.nodes[:2]], dtype=float)
            axis = coordinates[1] - coordinates[0]
            axis /= np.linalg.norm(axis)
            v = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        return np.concatenate([v, [self.J, len(p)], np.column_stack([p.y, p.z, p.area]).ravel()])

    def assignSectionPropertiesToElement(self, element, **kwargs):
        material = kwargs.get("material", self.material)

        if element.ensightType not in ("bar2", "bar3") or "rotation" not in element.fields[0]:
            raise Exception(f"Beam section is incompatible with element {element.elNumber} ({element.elType}).")

        element.setProperties(np.asarray(self.elementProperties(element), dtype=float))
        element.initializeElement()
        if not isinstance(material, dict):
            raise Exception(f"Beam section {self.name}: beam elements require a Marmot material.")
        element.setMaterial(material["name"], material["properties"])
