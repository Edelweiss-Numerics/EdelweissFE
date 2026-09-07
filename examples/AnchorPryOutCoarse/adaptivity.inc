**=================================================================================================
** LIVE h-ADAPTIVITY  --  shared by all four variants
**=================================================================================================
** The mesh refines DURING the analysis, wherever damage has started, and it does so identically in
** the implicit and explicit variants -- which is what makes the two comparable.
**
** DAMAGE, NOT STRESS, is the marked quantity, and that is not a free choice. Damage is monotone. An
** instantaneous stress threshold is defensible under implicit statics but wrong under explicit
** dynamics, where the stress carries the full wave content and there is no coarsening anywhere in
** the code, so the marked set ratchets towards refining everything.
**
** maxLevel=1 with splitFactor=2: each marked element splits once, into eight children. The marker
** still fires REPEATEDLY, because as the damage zone spreads it marks elements it had not marked
** before. Measured on the four variants of this example: 47 and 51 occasions implicitly, 12 in each
** explicit run (which checks only every topology-check-frequency increments), all four arriving at
** 2201-2215 concrete elements from the same starting 1634 -- the same refined mesh by different
** routes, which is what makes the four comparable.
**
** A refinement introduces a hanging-node transition into an already-damaged stress field, so damage
** near a transition cannot be cleanly separated from the transition itself. That is the price of
** live refinement; an initialOnly marker over a pry-out-cone-shaped element set avoids it at the
** cost of refining a region you have to guess in advance.
*modelModifier, type=hAdaptivity, name=amrConcrete
>>marker, type=fieldOutput, fieldOutput=damageForAMR, operator='>=', threshold=0.02
refineElSet=concrete
maxLevel=1
splitFactor=2
minMarkedElements=1
