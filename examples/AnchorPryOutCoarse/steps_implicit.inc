**=================================================================================================
** SOLVER AND STEPS -- implicit
**=================================================================================================
** linsolver=pardiso: a direct solver, chosen here so the example is reproducible on any machine
** without tuning. For a production-sized model, `blockamg` with the settings in
** examples/AnchorPryOut is far faster -- but its iterative state is not part of a restart, so a
** resumed run drifts from a continuous one under it, which would only confuse a template.
*solver, solver=NISTParallel, name=theSolver
linsolver=pardiso

** Step 1 -- a trivial warm-up at zero load. It costs one increment and settles the tie and contact
** state before any load is applied, which a Newton solve wants and an explicit run does not need
** (hence no counterpart in steps_explicit.inc).
*step, solver=theSolver
maxInc=1, minInc=1, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixLeft,        nSet=left,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=0

** Step 2 -- ramp the plate to 1.5 mm, the SAME final displacement the explicit variant reaches, so
** the two can be compared at equal load rather than at equal time.
**
** extrapolation=off is not cosmetic. With the linear predictor this model goes into a cutback
** spiral once contact and damage are both active; with the predictor off it marches steadily.
*step, solver=theSolver
maxInc=1e-2, minInc=1e-8, maxNumInc=1000, maxIter=25, criticalIter=10, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLeft,        nSet=left,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=1.5
