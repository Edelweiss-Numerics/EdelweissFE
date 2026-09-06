**=================================================================================================
** h-ADAPTIVITY -- ONE INITIAL PASS ON THE BOREHOLE, THEN LIVE REFINEMENT WHERE THE CONCRETE YIELDS
**=================================================================================================
** Two markers, doing two different jobs:
**
**   1. A SURFACE marker on 'concrete_to_mortar', initialOnly=True. That surface is the borehole
**      wall -- the concrete side of the concrete/mortar tie -- over the full 80 mm embedment. It is
**      evaluated once, before the first increment, and never again. This is the refinement that is
**      always done on this model: the load enters the concrete through that sleeve, and a coarse
**      borehole layer under-resolves the bearing stresses no matter what happens later.
**
**   2. A FIELD OUTPUT marker on 'alphaPForAMR' with '>= 1.0', evaluated for the whole run. That
**      field is declared in outputs.inc as GCDP's hardening variable maxed over the element's
**      quadrature points, and alphaP >= 1 is exactly "this element has left the elastic domain".
**      So the mesh follows the damage as it localizes, rather than being told in advance where the
**      localization is going to be.
**
** Why alphaP and not a stress, and not the damage omega: alphaP is MONOTONE. A stress threshold
** carries the full wave content of an explicit run and flickers, and because nothing in this code
** COARSENS, every flicker is permanent -- a flickering marker ratchets towards refining everything.
** omega would also be monotone, but it lags: it only starts moving once alphaP is already past 1.
**
** maxLevel=1 with splitFactor=2: a marked element splits into 8 children exactly once, so no
** element is refined twice, whichever marker claims it. splitFactor=3 is free for the implicit
** variant and is NOT free for the explicit one -- it takes the refined concrete below the steel's
** stable increment, which the explicit solver does not notice (see steps_explicit.inc). Holding
** both variants at 2 is also what makes their meshes comparable.
**
** minMarkedElements=10 suppresses a topology change raised for a handful of elements. Each one
** costs a re-lump of the mass, a rebuild of the hanging-node constraints and a re-pack of the DOF
** vector, which is not worth paying to refine three elements.
**
** refineElSet restricts the AMR octree to the concrete elset, the only hex20-family set here.
**
** VERIFY ON EVERY RUN: the first topology update should report
**
**   AMR ModelModifier: marked 110, refined -> active elements 11176 -> 11946, 574 hanging nodes
**
** that being marker 1 acting alone (alphaP is still 0 everywhere at t = 0). A marker whose named
** input is missing does NOT fail -- it marks nothing, and the run completes looking entirely normal
** on a mesh that never refined. If that line says 'marked 0', check that 'concrete_to_mortar'
** (model.inc) and 'alphaPForAMR' (outputs.inc) are spelled there as they are spelled here.
*modelModifier, type=hAdaptivity, name=amrConcreteBorehole
>>marker, type=surface, surface=concrete_to_mortar, initialOnly=True
>>marker, type=fieldOutput, fieldOutput=alphaPForAMR, operator='>=', threshold=1.0
refineElSet=concrete
maxLevel=1
splitFactor=2
minMarkedElements=10
