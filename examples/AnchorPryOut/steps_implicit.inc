**=================================================================================================
** SOLVER AND STEPS
**=================================================================================================
*solver, solver=NISTParallel, name=theSolver
linsolver=blockamg
linsolverConfigFile=blockamg.json

**=================================================================================================
** STEPS
**=================================================================================================
** Step 1 ("zero" in the source model): trivial warm-up step at zero load, establishing consistent
** tie/contact state before loading (mirrors the source model's zero-length *Static warm-up step).
*step, solver=theSolver
maxInc=1, minInc=1, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=0

** Step 2 ("loading" in the source model): ramp the plate's x-displacement to 5.0 mm using an
** Abaqus SMOOTH STEP-equivalent amplitude (quintic smoothstep, zero slope/curvature at both ends).
** Increment control mirrors the source model's *Dynamic data line (5e-3, 1, 1e-08, 5e-3).
*step, solver=theSolver
maxInc=5e-3, minInc=1e-8, maxNumInc=100000, maxIter=25, criticalIter=10, stepLength=1
>>options, name=theSolver, extrapolation=off, useAmgclMPCCondensation=True
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=5.0
