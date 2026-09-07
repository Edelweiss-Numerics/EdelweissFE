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
** The connectivity of a contact pair is the outcome of a search; this is how often that search is
** re-run. What makes 500 safe: 500 increments at the dt reported below is 8.6e-05 s of physical
** time, and the plate's peak speed under the quintic smoothstep is 1.875 * 5.0 mm / stepLength =
** 93.75 mm/s, so the surfaces move at most ~0.008 mm between searches -- under 1 % of the smallest
** facet edge (1.0 mm). The solver prints the largest nodal motion since the previous search on
** every update: CHECK IT on any run that changes stepLength, the mass scaling or the mesh. The
** margin above is not transferable, it is arithmetic on this file's numbers.
**
** Why this is an option at all: the search used to be an O(slaves x facets) closest-point loop in
** Python, 5.98 s per call on this model, which at a frequency of 100 cost as much per increment as
** the entire element pass. It now goes through a cKDTree broadphase with a per-facet prune, which
** took constraint connectivity on the reference run from 168 s to 11.5 s. 500 is therefore no
** longer a number tuned for cost -- it is tuned for accuracy, and can be lowered if the check
** above says the surfaces are outrunning it.
contact-update-frequency=500
report-performance=True
** How often the AMR markers are re-evaluated mid-run. NOT 0 here: adaptivity.inc carries a live
** fieldOutput marker (alphaP >= 1) alongside the initialOnly surface marker, so the mesh has to go
** on following the damage after the first increment. It must be a MULTIPLE of output-frequency,
** because a marker refines on the last FINALIZED field output -- 10000 is one reporting interval.
**
** Each check that actually refines costs a re-lump of the mass, a rebuild of the hanging-node
** constraints and a re-pack of the DOF vector. minMarkedElements in adaptivity.inc is what stops
** that being paid for a handful of elements.
topology-check-frequency=10000

**=================================================================================================
** STEPS
**=================================================================================================
** A single step: ramp the plate's x-displacement to 5.0 mm over stepLength seconds with a quintic
** smoothstep amplitude. The increment is not set here -- the solver computes courant-number *
** dt_crit and reports it at step start as "Critical time step for explicit dynamics". THE REPORTED
** NUMBER ALREADY HAS THE COURANT FACTOR IN IT; do not multiply by 0.1 again.
**
** MEASURED on this deck: 1.725589e-07 s. That is sqrt(10) above what v13/v15 of this model ran at,
** purely because every density in materials_explicit.inc is scaled by 100 rather than 10. The ramp
** therefore needs stepLength / dt = 1e-1 / 1.725589e-07 = about 580 000 increments, and
** maxNumInc=2500000 is roughly 4x that -- deliberately generous, because reaching maxNumInc ends
** the step NORMALLY and silently short of 5.0 mm rather than reporting a failure. To turn this file
** into a smoke test, lower maxNumInc to a few hundred.
**
** The STEEL governs dt, not the concrete: the value above was unchanged across all eight topology
** changes of the reference run, from 11176 to 14144 active elements, so the borehole refinement
** costs no extra increments. dt IS re-estimated after every topology change and lowered if
** refinement tightened the limit -- the solver says so ("Refinement lowered the stable time
** increment from ... to ...") -- so a refinement that merely shrinks an element is handled.
**
** WHAT IS NOT HANDLED, and it is worth knowing before running this for a day: that estimate is a
** minimum over ELEMENTS, and no element-based estimate can see a NODE whose lumped mass is a small
** fraction of its neighbours'. Refining at a sliver produces exactly such a node. In the reference
** run the lightest node sat at 1.1e-03 of the median lumped mass from the FIRST refinement onwards
** and the run integrated happily for 130 000 increments; the eighth refinement made that node 7 %
** lighter again, and the solution passed 1e+97 within the next 500 increments -- with dt duly
** re-estimated at every one of those topology changes and never once found wanting.
**
** The solver already prints the quantity that predicts this, on every topology change:
**
**   smallest integrating mass 1.472e-07 (1.1e-03 of median)
**
** Nothing acts on it. WATCH THAT RATIO; a run whose lightest node is orders below the median is
** living on margin, however healthy the mass, momentum and energy lines next to it look. It is also
** why splitFactor stays at 2 and maxLevel at 1.
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=2500000, maxIter=25, stepLength=1e-1
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=5.0, f(t)='6*t**5 - 15*t**4 + 10*t**3'
