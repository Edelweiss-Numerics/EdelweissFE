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

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.facetcontactgeometry import (
    line2GapGradientHessian,
    tria3GapGradientHessian,
)
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A penalty based unilateral contact constraint between a node set of ordinary FE nodes and a
deformable master surface represented by flat contact facet elements (:mod:`~edelweissfe.elements.
contactsurfaceelement`, typically created via :mod:`~edelweissfe.generators.surfaceelementgenerator`).
"""

module = Module(
    "nodeToDeformableSurfacePenalty",
    "A penalty based unilateral contact constraint preventing nodes of a node set from penetrating "
    "a deformable surface represented by contact facet elements.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("nSet", "The (slave) node set to be protected from penetrating the surface.", str)
module.addRequiredArg(
    "surface", "The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet).", str
)
module.addRequiredArg("penalty", "The numerical penalty value.", float)

module.addOptionalArg(
    "type",
    "The formulation type: 'linear' (linear force, constant stiffness with jump) or 'quadratic' (quadratic "
    "force, linear stiffness).",
    str,
    "linear",
)
module.addOptionalArg(
    "searchDistance",
    "An optional broadphase distance for the per-increment candidate-facet search. If not given, "
    "every slave is always assigned its single closest facet, without a distance gate.",
    float,
    None,
)

documentation = [module]


class DeformableSurfaceContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of
    :class:`Constraint`.

    Each currently-active slave couples only to its own self-block and its currently-assigned
    facet's self-block and slave-facet coupling blocks -- there is no coupling between different
    slave nodes, nor between a slave and any facet it is not currently assigned to.

    Attributes
    ----------
    K_pp : list[numpy.ndarray]
        List of per-slave views of shape ``(nDim, nDim)``, the self-block of each slave node.
    K_ff : list[numpy.ndarray]
        List of per-slave views of shape ``(m, m)``, the self-block of the assigned facet
        (``m = nFacetNodes * nDim``).
    K_pf : list[numpy.ndarray]
        List of per-slave views of shape ``(nDim, m)``, slave-to-facet coupling.
    K_fp : list[numpy.ndarray]
        List of per-slave views of shape ``(m, nDim)``, facet-to-slave coupling (transpose of
        ``K_pf``).
    """

    def __init__(self, flat_array: np.ndarray, nDim: int, facetNodeCounts: list[int]):
        self.K_pp = []
        self.K_ff = []
        self.K_pf = []
        self.K_fp = []

        offset = 0
        for nFacetNodes in facetNodeCounts:
            m = nFacetNodes * nDim

            pp = flat_array[offset : offset + nDim * nDim].reshape((nDim, nDim))
            offset += nDim * nDim

            ff = flat_array[offset : offset + m * m].reshape((m, m))
            offset += m * m

            pf = flat_array[offset : offset + nDim * m].reshape((nDim, m))
            offset += nDim * m

            fp = flat_array[offset : offset + m * nDim].reshape((m, nDim))
            offset += m * nDim

            self.K_pp.append(pp)
            self.K_ff.append(ff)
            self.K_pf.append(pf)
            self.K_fp.append(fp)


def _tria3Containment(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[float, float, bool]:
    """Barycentric-like in-plane coordinates (alpha, beta) of the projection of xs onto the
    (possibly non-orthogonal) basis spanned by (x2-x1, x3-x1), and whether that projection falls
    inside the triangle."""

    e1 = x2 - x1
    e2 = x3 - x1
    r = xs - x1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    rTangential = r - r.dot(n) * n

    A = np.array([[e1.dot(e1), e1.dot(e2)], [e1.dot(e2), e2.dot(e2)]])
    b = np.array([e1.dot(rTangential), e2.dot(rTangential)])
    alpha, beta = np.linalg.solve(A, b)

    inside = alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0
    return alpha, beta, inside


def _line2Containment(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, bool]:
    """Parametric coordinate t of the projection of xs onto the edge (x1,x2), and whether that
    projection falls inside the segment."""

    e = x2 - x1
    t = (xs - x1).dot(e) / e.dot(e)
    return t, 0.0 <= t <= 1.0


class Constraint(ConstraintBase):
    """
    Penalty based unilateral contact between a slave node set and a deformable master surface
    represented by flat (Tria3/Line2) contact facet elements.

    Theoretical background
    -----------------------
    Each facet is exactly flat (a plane through 3 nodes, or a line through 2 nodes), so its normal
    has exactly zero curvature over its own domain -- mirroring why the discrete rigid body
    contact's triangulated master surface didn't need a curvature term either. Unlike that rigid
    case, though, each facet's nodes are ordinary displacement DOFs of a deformable body, and
    different facets have disjoint DOFs -- so the set of candidate master facets must be kept in
    sync with the actual equation system rather than fixed once for the whole analysis.

    This constraint implements :meth:`updateConnectivity`, called once per increment (before the
    equation system is (re)built, see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`),
    re-assigning each slave node to its single closest facet (within ``searchDistance``, if given)
    based on the last converged configuration -- mirroring the pattern already used by
    EdelweissMeshfree's ``NonlinearQuasistaticSolver``/``DiscreteRigidBodyPenaltyContact`` for
    dynamic contact-pair connectivity. Within :meth:`applyConstraint`, the gap, its exact gradient,
    and its exact Hessian (see :mod:`~edelweissfe.utils.facetcontactgeometry`, including the
    second-derivative term from the facet normal's own pose-dependence -- not curvature, since the
    facet is flat) are recomputed fresh from the *current Newton iterate* every iteration, exactly
    as :mod:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty` does for rigid bodies -- no
    geometry is frozen across iterations within an increment.

    Each slave is assigned at most *one* active facet at a time (reassigned each increment); this
    is a deliberate simplification relative to a multi-candidate-per-slave design -- see the
    project plan this was built from for the more elaborate alternative and why it was not needed
    here. If the slave's assigned facet ever fails its exact in-facet containment test mid-Newton
    (the true contact point has moved onto a neighboring facet within the same increment), no
    contact contribution is assembled for that slave until the next connectivity update -- the
    same accepted non-smoothness at facet boundaries as the rigid-body case's mesh edges.

    Currently only available for spatialdomain = 3D (Tria3 facets) or 2D (Line2 facets), matching
    whichever facet type populates the given ``surface`` element set.
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)

        kwargs = CaseInsensitiveDict(kwargs)

        self.slaveNodes = list(model.nodeSets[kwargs["nSet"]])
        self.facetElements = list(model.elementSets[kwargs["surface"]])
        self.nSlaves = len(self.slaveNodes)

        self.penalty = kwargs["penalty"]
        self.type = kwargs["type"].lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")
        self.searchDistance = kwargs["searchDistance"]

        self.nDim = model.domainSize

        self._referenceCoordsSlaves = np.array([n.coordinates for n in self.slaveNodes])
        self._referenceCoordsFacets = [np.array([n.coordinates for n in el.nodes]) for el in self.facetElements]

        self._assignedFacetIdx = [None] * self.nSlaves

        self._nodes = []
        self._fieldsOnNodes = []
        self._nDof = 0

        self.totalNormalForce = 0.0

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def _currentCoordinates(self, nodes: list, model: FEModel, referenceCoords: np.ndarray) -> np.ndarray:
        dispField = model.nodeFields.get("displacement")
        if dispField is None or "U" not in dispField:
            return referenceCoords.copy()
        idcs = dispField._indicesOfNodesInArray
        u = np.array([dispField["U"][idcs[n]] if n in idcs else np.zeros(self.nDim) for n in nodes])
        return referenceCoords + u

    def updateConnectivity(self, model: FEModel) -> bool:
        """Re-assign each slave node to its single closest facet, based on the last converged
        configuration. Called once per increment by the solver, before the equation system is
        (re)built."""

        slaveCoords = self._currentCoordinates(self.slaveNodes, model, self._referenceCoordsSlaves)
        facetCentroids = np.array(
            [
                np.mean(self._currentCoordinates(el.nodes, model, self._referenceCoordsFacets[i]), axis=0)
                for i, el in enumerate(self.facetElements)
            ]
        )

        newAssignment = [None] * self.nSlaves
        for s in range(self.nSlaves):
            distances = np.linalg.norm(facetCentroids - slaveCoords[s], axis=1)
            closest = int(np.argmin(distances))
            if self.searchDistance is None or distances[closest] <= self.searchDistance:
                newAssignment[s] = closest

        hasChanged = newAssignment != self._assignedFacetIdx
        self._assignedFacetIdx = newAssignment

        newNodes = []
        newFieldsOnNodes = []
        for s in range(self.nSlaves):
            newNodes.append(self.slaveNodes[s])
            newFieldsOnNodes.append(["displacement"])
            if newAssignment[s] is not None:
                facetNodes = self.facetElements[newAssignment[s]].nodes
                newNodes.extend(facetNodes)
                newFieldsOnNodes.extend([["displacement"]] * len(facetNodes))

        if newNodes != self._nodes:
            hasChanged = True

        self._nodes = newNodes
        self._fieldsOnNodes = newFieldsOnNodes
        self._nDof = sum(self.nDim for _ in newNodes)

        return hasChanged

    def getVIJContributionSize(self) -> int:
        size = 0
        for s in range(self.nSlaves):
            if self._assignedFacetIdx[s] is None:
                continue
            m = len(self.facetElements[self._assignedFacetIdx[s]].nodes) * self.nDim
            size += self.nDim**2 + m * m + 2 * self.nDim * m
        return size

    def shapeVIJContribution(self, flat_view: np.ndarray) -> DeformableSurfaceContactStiffnessView:
        facetNodeCounts = [
            len(self.facetElements[self._assignedFacetIdx[s]].nodes)
            for s in range(self.nSlaves)
            if self._assignedFacetIdx[s] is not None
        ]
        return DeformableSurfaceContactStiffnessView(flat_view, nDim=self.nDim, facetNodeCounts=facetNodeCounts)

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        k = offset
        localOffset = 0
        for s in range(self.nSlaves):
            pIdcs = [idcs[localOffset + i] for i in range(self.nDim)]
            localOffset += self.nDim

            if self._assignedFacetIdx[s] is None:
                continue

            nFacetNodes = len(self.facetElements[self._assignedFacetIdx[s]].nodes)
            m = nFacetNodes * self.nDim
            fIdcs = [idcs[localOffset + i] for i in range(m)]
            localOffset += m

            for i in range(self.nDim):
                for j in range(self.nDim):
                    I_[k] = pIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

            for i in range(m):
                for j in range(m):
                    I_[k] = fIdcs[i]
                    J_[k] = fIdcs[j]
                    k += 1

            for i in range(self.nDim):
                for j in range(m):
                    I_[k] = pIdcs[i]
                    J_[k] = fIdcs[j]
                    k += 1

            for i in range(m):
                for j in range(self.nDim):
                    I_[k] = fIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: DeformableSurfaceContactStiffnessView,
        timeStep: TimeStep,
    ):
        self.totalNormalForce = 0.0

        localOffset = 0
        activeIdx = 0
        for s in range(self.nSlaves):
            pStart = localOffset
            localOffset += self.nDim

            if self._assignedFacetIdx[s] is None:
                continue

            facetElement = self.facetElements[self._assignedFacetIdx[s]]
            nFacetNodes = len(facetElement.nodes)
            m = nFacetNodes * self.nDim
            fStart = localOffset
            localOffset += m

            pIdcs = list(range(pStart, pStart + self.nDim))
            fIdcs = list(range(fStart, fStart + m))

            xs = self._referenceCoordsSlaves[s] + U_np[pIdcs]
            facetU = U_np[fIdcs].reshape((nFacetNodes, self.nDim))
            facetCoords = self._referenceCoordsFacets[self._assignedFacetIdx[s]] + facetU

            if nFacetNodes == 3:
                alpha, beta, inside = _tria3Containment(xs, *facetCoords)
                if not inside:
                    activeIdx += 1
                    continue
                g, w, H = tria3GapGradientHessian(xs, *facetCoords)
            else:
                t, inside = _line2Containment(xs, *facetCoords)
                if not inside:
                    activeIdx += 1
                    continue
                g, w, H = line2GapGradientHessian(xs, *facetCoords)

            if g >= 0.0:
                activeIdx += 1
                continue

            if self.type == "linear":
                f_n = self.penalty * g
                stiffness = self.penalty
            else:
                f_n = 0.5 * self.penalty * g**2
                stiffness = self.penalty * g

            globalIdcs = pIdcs + fIdcs
            PExt[globalIdcs] -= f_n * w

            K.K_pp[activeIdx] += (
                stiffness * np.outer(w[: self.nDim], w[: self.nDim]) + f_n * H[: self.nDim, : self.nDim]
            )
            K.K_ff[activeIdx] += (
                stiffness * np.outer(w[self.nDim :], w[self.nDim :]) + f_n * H[self.nDim :, self.nDim :]
            )
            K.K_pf[activeIdx] += (
                stiffness * np.outer(w[: self.nDim], w[self.nDim :]) + f_n * H[: self.nDim, self.nDim :]
            )
            K.K_fp[activeIdx] += (
                stiffness * np.outer(w[self.nDim :], w[: self.nDim]) + f_n * H[self.nDim :, : self.nDim]
            )

            self.totalNormalForce += f_n
            activeIdx += 1
