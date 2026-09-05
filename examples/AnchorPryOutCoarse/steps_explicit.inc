**=================================================================================================
** SOLVER AND STEP -- explicit
**=================================================================================================
** COURANT NUMBER 0.1 -- the expensive decision in this file. READ THIS BEFORE RAISING IT.
**
** The solver's critical increment l/c is a LINEAR-element formula: the stable increment for an
** element whose mass is spread uniformly over its nodes and whose highest eigenfrequency is c/l. A
** 20-node element satisfies neither. The mass part is handled in Marmot (HRZ lumping gives a hexa20
** corner half the uniform share, so the estimate carries a 0.7071 factor); the polynomial-order part
** is NOT, and a quadratic element's highest free eigenfrequency is several times what l/c predicts.
**
** Measured on testfiles/marmot/GCDPNEDExplicit -- same element and material as this concrete --
** final reaction against courant number:
**
**     0.8 -> 4.49      0.4 -> 84.6      0.2 -> 118.5      0.1 -> 120.7      0.05 -> 118.0
**
** It converges at 0.1-0.2. At the 0.8 default it returns a number 26x too small AND DOES NOT FAIL:
** the run completes, the reaction curve looks smooth, and it is wrong. The divergence is also only
** visible on a material with no damping -- GC3D20R with gcdp sat above the stability limit for 800
** increments looking healthy, because the nonlocal viscosity absorbed it.
*solver, solver=NEDParallel, name=theSolver
second-order-fields="displacement"
first-order-fields="nonlocal damage"
courant-number=0.1
output-frequency=1000
** How often the contact constraints re-run their closest-point search. Throttling is what makes
** contact affordable here: the search is O(slaves x facets) in Python, which is fine once per
** increment of an implicit analysis and not fine a hundred thousand times. What makes it safe rather
** than merely cheap is that an explicit increment is tiny -- the solver reports the largest nodal
** motion since the previous search at every connectivity change, and it stays orders of magnitude
** below a facet dimension. CHECK THAT REPORTED MOTION if you change stepLength or the mesh.
contact-update-frequency=500
** Must be a non-zero multiple of output-frequency: a marker refines on the last FINALIZED field
** output, and the solver refuses the combination otherwise.
topology-check-frequency=5000
** RAISED from the 1e-6 default, and this is a measurement rather than a workaround.
**
** The solver asserts that a topology change conserves the total lumped mass. That mass IS a
** geometric identity -- children tile the parent and carry its density -- but it is ASSEMBLED by
** Gauss quadrature, which is exact only to a polynomial order, and a distorted hexa20 has a
** non-polynomial Jacobian. Repartitioning a parent into children therefore changes the quadrature
** truncation error, by an amount that depends on the mesh.
**
** The 1e-6 default is calibrated on the production pry-out mesh, where one refinement moves the
** total mass by 4.6e-08. This mesh is 2x coarser and its refinements move it by 2.2e-06 -- 48x more,
** which is how a truncation-error term behaves under coarsening, and still five orders below the
** O(1) that a genuine refinement or lumping error would produce. Without this line the run aborts at
** its first refinement.
**
** Do not copy this number onto a different mesh without re-measuring, and never raise it to silence
** a violation you have not first shown to scale with h.
mass-conservation-tolerance=1e-5

** A single step: ramp the plate to 1.5 mm -- the SAME final displacement as the implicit variant.
**
** stepLength is PHYSICAL SECONDS here, not the implicit model's dimensionless 1, and together with
** the mass scaling in materials_explicit.inc it is what sets the cost: the solver enforces
** courant-number * dt_crit and reports it at step start, giving dt ~ 3.4e-7 s, so this 0.024 s ramp
** is about 70 000 increments. maxNumInc is generous; reaching it early is not a failure, the step
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
