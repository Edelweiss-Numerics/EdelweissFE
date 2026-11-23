**=================================================================================================
** SOLVER AND STEP -- explicit
**=================================================================================================
*solver, solver=NEDParallel, name=theSolver
** The time-integration scheme of every field is DERIVED from the assembled inertia and damping
** (#134); this line only ASSERTS the expected outcome and aborts the step if it differs. Both fields
** are second order: GCDP's micro-inertia m_k (property 21, materials_explicit.inc) gives the nonlocal
** damage field an inertia, so it is integrated by central difference. The assertion catches a deck
** that loses m_k -- the property is optional and silently defaults to zero.
expect-second-order-fields="displacement, nonlocal damage"
output-frequency=125
** How often the contact constraints re-run their closest-point search. Throttling is what makes
** contact affordable here: the search is O(slaves x facets) in Python, which is fine once per
** increment of an implicit analysis and not fine a hundred thousand times. What makes it safe rather
** than merely cheap is that an explicit increment is tiny -- the solver reports the largest nodal
** motion since the previous search at every connectivity change, and it stays orders of magnitude
** below a facet dimension. CHECK THAT REPORTED MOTION if you change stepLength or the mesh. The three
** cadences here are counted in increments, so they follow the time increment: output, contact search
** and topology check every 125, 60 and 625 increments keep the physical intervals of the model.
contact-update-frequency=60
** Must be a non-zero multiple of output-frequency: a marker refines on the last FINALIZED field
** output, and the solver refuses the combination otherwise.
topology-check-frequency=625
** RAISED from the 1e-6 default, and this is a measurement rather than a workaround.
**
** The solver asserts that a topology change conserves the total lumped mass. That mass IS a
** geometric identity -- children tile the parent and carry its density -- but it is ASSEMBLED by
** Gauss quadrature, which is exact only to a polynomial order, and a distorted hexa20 has a
** non-polynomial Jacobian. Repartitioning a parent into children therefore changes the quadrature
** truncation error, by an amount that depends on the mesh.
**
** The 1e-6 default is calibrated on the production pry-out mesh, where one refinement moves the
** total mass by 4.63e-08. This mesh is 2x coarser and its first refinement moves it by 2.217e-06 --
** 48x more, which is how a truncation-error term behaves under coarsening, and still five orders
** below the O(1) that a genuine refinement or lumping error would produce.
**
** Both halves of that were measured by deleting this line and running: the abort arrives at the
** first topology change (1634 -> 1648 active elements, increment 10000) and names the figure --
** "2.718504e+01 became 2.718510e+01, a relative change of 2.217260e-06 against a tolerance of
** 1.000000e-06".
**
** Do not copy this number onto a different mesh without re-measuring, and never raise it to silence
** a violation you have not first shown to scale with h.
lumped-quantity-conservation-tolerance=1e-5

** A single step: ramp the plate to 1.5 mm -- the SAME final displacement as the implicit variant.
**
** stepLength is PHYSICAL SECONDS here, not the implicit model's dimensionless 1, and together with
** the mass scaling in materials_explicit.inc it is what sets the cost: the solver enforces
** courant-number * dt_crit (the elements' eigenvalue bound) and reports it at step start, giving
** dt ~ 2.7e-6 s, ~1.7e-6 s after the first refinement, so this 0.024 s ramp is about 11 600 increments. maxNumInc is generous; reaching it early is not a failure, the step
** simply ends.
**
** f(t)='t**2' starts the plate from rest with zero velocity, so the structure is not shocked at
** t = 0. The reaction and displacement histories are what decide whether the result is usable, not
** the fact that it completed -- see plot_damage_comparison.py, which reports the kinetic energy as a
** fraction of the external work.
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=200000, maxIter=25, stepLength=2.40e-2
>>dirichlet, name=fixLeft,        nSet=left,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,            field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=1.5, f(t)='t**2'
