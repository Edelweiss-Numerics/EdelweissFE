**=================================================================================================
** SOLVER AND STEPS
**=================================================================================================
*solver, solver=NEDParallel, name=theSolver
second-order-fields="displacement"
first-order-fields="nonlocal damage"
** COURANT NUMBER: 0.1, and this is the expensive decision in this file -- READ THIS BEFORE RAISING IT.
**
** The solver's critical time step is l / c, which is a LINEAR-element formula: it is the stable
** increment for an element whose mass is spread uniformly over its nodes and whose highest
** eigenfrequency is c/l. A 20-node element satisfies neither. The mass part is now handled in Marmot
** (HRZ lumping gives a hexa20 corner half the uniform share, so the estimate carries a 0.7071
** factor), but the polynomial-order part is NOT: a quadratic element's highest free eigenfrequency is
** several times what l / c predicts.
**
** Measured on testfiles/marmot/GCDPNEDExplicit, same element and material as this model's concrete,
** final reaction against courant number:
**
**     0.8 -> 4.49      0.4 -> 84.6      0.2 -> 118.5      0.1 -> 120.7      0.05 -> 118.0
**
** It converges at 0.1-0.2. At the 0.8 default it returns a number 26x too small AND DOES NOT FAIL --
** the run completes, the reaction curve looks smooth, and it is wrong. Note also that the divergence
** only becomes visible at all on a material with no damping: GC3D20R + gcdp sat above the stability
** limit for 800 increments looking healthy because the nonlocal viscosity absorbed it.
**
** 0.1 makes this run roughly 5x more expensive than the 0.5 an earlier version of this file used
** (~11 h rather than ~2.2 h at 64 threads). That is the cost of an answer that does not depend on the
** time step. Do not raise it without repeating the study above on THIS mesh.
courant-number=0.1
output-frequency=10000
** MEASURED on this model (500-increment smoke test, 16 threads, 280155 dof): one connectivity
** search costs 5.98 s, because nodeToDeformableSurfacePenalty.updateConnectivity is an
** O(slaves x facets) closest-point loop in Python. At a frequency of 100 that is 0.06 s amortised
** per increment -- as much as the entire element pass (0.058 s) -- and about 1.5 h over a full run.
** 500 brings that to 0.29 h.
**
** What makes 500 safe rather than merely cheaper: the solver reports the largest nodal motion since
** the previous search on every change. In the smoke test that was 7e-8 to 3e-6 mm against a smallest
** facet edge of 1.0 mm. At the steepest part of the smoothstep the plate moves at 1.875 * 5 mm /
** stepLength, so at stepLength=1e-2 a 500-increment window is ~0.05 mm, still 5 % of a facet. CHECK
** THE REPORTED MOTION on any run that changes stepLength or the mesh -- do not carry this number
** over blindly.
**
** The real fix is to give that search a spatial index; the tie constraint's reconcile() had the
** identical O(n*m) shape and a cKDTree took it 65-232x faster.
contact-update-frequency=500
report-performance=True
** 0 = run the topology update exactly once, before the increment loop, and never again. Every
** marker in this file is initialOnly, so hAdaptivity.actsOnlyAtSimulationStart is True and the
** solver would not schedule a later update anyway -- 0 states the intent explicitly instead of
** leaving it to be inferred. (A non-zero value must be a multiple of output-frequency, because a
** marker refines on the last FINALIZED field output.)
topology-check-frequency=10000

**=================================================================================================
** STEPS
**=================================================================================================
** A single step: ramp the plate's x-displacement to 5.0 mm over stepLength seconds with a quintic
** smoothstep amplitude. The time increment is not set here -- the solver enforces
** courant-number * dt_crit, which it reports at step start; cross-check it against the 2.45e-7 s
** the mesh implies (see header note 5) before trusting a long run.
**
** maxNumInc is what bounds this file: at 500 it is a smoke test, and it has to be raised to
** stepLength / dt_crit for a real run. Reaching it is not a failure -- the step ends normally.
**
** MEASURED at courant 0.5: dt_crit = 1.146058e-07 s, so stepLength=1e-2 needs maxNumInc=87300.
** That is 6.5 % tighter than the 1.226e-07 the mesh survey predicted, because the governing element
** is distorted and Marmot takes the minimum of cbrt(8*detJ) over its quadrature points rather than
** the cube root of its volume. The steel elements govern, as predicted -- the refined concrete sits
** at 3.41e-07 -- so the initial refinement below genuinely costs no extra increments.
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=2500000, maxIter=25, stepLength=1e-1
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=5.0, f(t)='6*t**5 - 15*t**4 + 10*t**3'
