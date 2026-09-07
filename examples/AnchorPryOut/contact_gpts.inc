**=================================================================================================
** CONTACT (Abaqus General Contact, linear pressure-overclosure -> penalty=1e6, type=linear)
**=================================================================================================
** V15 CHANGE, AND THE ONLY ONE: both contacts use the INTEGRATED (Gauss-point-to-segment)
** formulation instead of the node-based one. See issue #112. The node-based constraint cannot
** transmit a correct pressure across a hexa20 face at all: the consistent nodal load of a uniform
** pressure at a quad8 corner is TENSILE (-pA/12), a unilateral spring can only push, so the
** discrete solution opens every corner gap instead -- on the matched hexa20 patch test 0 of 9
** corner nodes carry load, the resultant is 1.7% short and the interface visibly undulates, which
** is the surface-spike signature seen in these pry-out runs. Integrating the pressure at quadrature
** points and distributing it with the parent face's own quadratic shape functions removes the
** obstruction (patch test: force and flatness exact to machine precision, corner nodal force equal
** to the analytic -4pA/12).
**
** penalty, sliding and type are UNCHANGED from v14, so the constitutive and kinematic setup is
** identical and the two runs are directly comparable. nQuadraturePoints=3 is the default and is the
** rule that integrates a quadratic face's shape functions exactly over a facet.
**
** plate/concrete bearing surface: Abaqus SMALL SLIDING -> sliding=small
*constraint, name=contactPlateConcrete, type=surfaceToDeformableSurfacePenalty
slaveSurface=plate_bottom_facets, masterSurface=concrete_top_facets, penalty=1e5, sliding=small, type=quadratic, nQuadraturePoints=3

** anchor shaft / plate hole
*constraint, name=contactShaftHole, type=surfaceToDeformableSurfacePenalty
slaveSurface=anchor_to_plate_facets, masterSurface=plate_to_anchor_facets, penalty=1e5, sliding=small, type=quadratic, nQuadraturePoints=3
