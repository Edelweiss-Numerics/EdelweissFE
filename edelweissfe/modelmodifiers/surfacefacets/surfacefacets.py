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


"""Regenerates contact/tie facet elements after the mesh underneath them changes.

Facet generation *is* a topology change -- it creates and deletes elements -- so it belongs in the
topology-update phase, alongside the modifiers whose refinements it reacts to. It used to hang off
the consumers instead: a tie regenerated its own facets from inside its reconcile. That made every
tie and contact a mutating consumer, which is what forced the old push-notification path and its
re-entrancy (a consumer minting elements while a modifier was still inside its own loop). With this
modifier owning the work, constraints become pure readers of ``model.elementSets[...]``.

The modifier is implicit: it is created automatically once any ``*surface`` facet recipe has been
registered (see :func:`~edelweissfe.generators.surfaceelementgenerator.buildContactFacets`), and is
ordered last so it reacts to whatever the primary modifiers did in the same round. Users do not
declare it -- they already declared the recipe.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.base.modelmodifierbase import ModelModifierBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType


@dataclass(frozen=True)
class FacetPlan:
    """The recipes to regenerate, by facet element set name.

    Recipe names rather than element numbers: the facets themselves are an *output* of applying this
    plan, and their numbers come from the model's allocator. What the decision actually consists of
    is "which surfaces need retiling", which is stable across a restart replay.
    """

    recipeNames: tuple

    def __init__(self, recipeNames):
        object.__setattr__(self, "recipeNames", tuple(str(name) for name in recipeNames))


class ModelModifier(ModelModifierBase):
    """Retiles contact/tie surfaces whose source elements changed."""

    #: Purely reactive: plan() returns None unless another modifier already changed the mesh.
    initiatesTopologyChanges = False

    def __init__(self, name: str, model: FEModel, journal: Journal, **kwargs):
        super().__init__(name, model, journal, **kwargs)

    def plan(self, model: FEModel, change, step, timeStep: float) -> "FacetPlan | None":
        """Retile every recorded recipe whose surface this change touched.

        On the first round of an update (``change is None``) there is nothing to react to yet: the
        facets already tile the mesh as it stands. Returning ``None`` there is also what keeps the
        pipeline from looping, since this modifier's own output would otherwise look like a reason
        to run again.
        """

        if change is None:
            return None

        touched = [
            facetSetName
            for facetSetName, recipe in model.contactFacetRecipes.items()
            if change.touchesSurface(recipe[0])
        ]
        return FacetPlan(recipeNames=sorted(touched)) if touched else None

    def apply(self, model: FEModel, plan: "FacetPlan") -> ModelChange:
        """Rebuild the named recipes' facets.

        Reads no solution state: which surfaces to retile is entirely in the plan, and the facet
        geometry follows from the current mesh -- so a replay reproduces this exactly.
        """

        change = ModelChange(kind=ModelChangeType.TOPOLOGY_CHANGE)
        for facetSetName in plan.recipeNames:
            recipe = model.contactFacetRecipes.get(facetSetName)
            if recipe is None:
                continue
            before = {el.elNumber for el in model.elementSets.get(facetSetName, [])}
            buildContactFacets(model, *recipe, self._journal)
            after = {el.elNumber for el in model.elementSets[facetSetName]}
            change.removedElements |= before - after
            change.addedElements |= after - before
            change.changedElementSets.add(facetSetName)
        return change

    def encodePlan(self, plan: "FacetPlan") -> dict:
        """Serialize the recipe names. h5py stores these as variable-length strings."""

        return {"recipeNames": np.array(list(plan.recipeNames), dtype=object)}

    def decodePlan(self, data: dict) -> "FacetPlan":
        """Inverse of :meth:`encodePlan`, tolerating the bytes h5py hands back."""

        return FacetPlan(
            recipeNames=[name.decode() if isinstance(name, bytes) else str(name) for name in data["recipeNames"]]
        )

    def declaredDomain(self, model: FEModel) -> set:
        """The facet elements this modifier owns -- every element of every recipe's facet set."""

        owned = set()
        for facetSetName in model.contactFacetRecipes:
            owned |= {el.elNumber for el in model.elementSets.get(facetSetName, [])}
        return owned
