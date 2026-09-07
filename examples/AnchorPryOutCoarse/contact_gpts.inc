**=================================================================================================
** CONTACT -- GPTS, Gauss-point-to-segment (integrated)
**=================================================================================================
** The same two interfaces, same penalty, same small-sliding kinematics as contact_nts.inc -- only
** the constraint type and nQuadraturePoints differ, which is the whole point of having both: swap
** this include for the other and nothing else about the model changes.
**
** Contact is enforced at quadrature points on the slave facets, and the resulting force is
** distributed with the PARENT face's quadratic shape functions. That is what lets a serendipity
** corner carry the tensile load its face demands, instead of opening a gap the way a per-node
** unilateral spring must. On a matched hexa20 patch test the node-based formulation loses 1.7 % of
** the interface force and lifts every corner off; the integrated one reproduces the exact resultant
** to 13 digits.
**
** nQuadraturePoints=3 is the default and is stated here because it is load-bearing: three points
** integrate the parent face's shape functions exactly over a Tria3 facet, which is what makes those
** consistent nodal loads -- including the negative corner loads -- come out exactly.
**
** Cost: the element block goes from 12 to 48 DOF with two to three times more points, so contact
** assembly is substantially heavier than NTS. On this model that is a small share of the total.

*constraint, name=contactPlateConcrete, type=surfaceToDeformableSurfacePenalty
slaveSurface=plate_bottom_facets, masterSurface=concrete_top_facets, penalty=1e5, sliding=small, type=quadratic, nQuadraturePoints=3

*constraint, name=contactShaftHole, type=surfaceToDeformableSurfacePenalty
slaveSurface=anchor_to_plate_facets, masterSurface=plate_to_anchor_facets, penalty=1e5, sliding=small, type=quadratic, nQuadraturePoints=3
