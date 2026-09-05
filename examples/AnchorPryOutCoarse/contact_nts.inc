**=================================================================================================
** CONTACT -- NTS, node-to-surface
**=================================================================================================
** Two genuinely unilateral interfaces: the fixture plate bearing on the concrete surface, and the
** anchor shaft inside the plate's clearance hole. Everything else in the model is tied (model.inc).
**
** Each slave NODE is projected onto the master facet mesh and carries one unilateral penalty spring.
** This is the classical formulation and the cheaper of the two.
**
** Its limitation shows on QUADRATIC faces, and it is worth knowing before choosing: the consistent
** nodal loads of a uniform pressure on a hexa20 face are NEGATIVE at the corners (-pA/12) and
** positive at the midsides (+pA/3). A unilateral per-node spring cannot pull, so the corners open
** instead, and the interface picks up a spurious compliance. Swap in contact_gpts.inc to see the
** difference -- on this model both master surfaces are hexa20 concrete or plate faces.
**
** sliding=small: the contact pairing is frozen per increment, which suits the small relative motion
** here and is what the GPTS variant requires (it refuses sliding=finite rather than approximating).

*constraint, name=contactPlateConcrete, type=nodeToDeformableSurfacePenalty
slaveSurface=plate_bottom_facets, masterSurface=concrete_top_facets, penalty=1e5, sliding=small, type=quadratic

*constraint, name=contactShaftHole, type=nodeToDeformableSurfacePenalty
slaveSurface=anchor_to_plate_facets, masterSurface=plate_to_anchor_facets, penalty=1e5, sliding=small, type=quadratic
