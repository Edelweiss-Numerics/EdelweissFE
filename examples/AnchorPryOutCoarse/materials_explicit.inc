**=================================================================================================
** MATERIALS -- explicit
**=================================================================================================
** Same model and same parameters as materials_implicit.inc, with three differences that an explicit
** integrator forces, plus two rate terms a static solve has no use for: the hyperbolic nonlocal
** field (GCDP idx 20/21) and the bulk viscosity at the end of this file. CONSISTENT UNITS: N, mm, MPa, tonne, s.
**
**  1. DENSITIES ARE REQUIRED. The solver builds a lumped mass from getDensity() on every element.
**
**  2. MASS SCALING x100. Every density is 100x its physical value (concrete 3e-7 instead of 3e-9).
**     The stable increment goes with 1/sqrt(rho), so this buys a factor of 10 and is what makes the
**     run reach the same 1.5 mm of plate displacement as the implicit variant in a comparable time.
**     The cost is that inertial forces at a given loading rate are 100x larger, which works AGAINST
**     quasi-staticity. That is not a matter of opinion: plot_damage_comparison.py integrates the
**     reaction against the displacement and reports the kinetic energy as a fraction of that work.
**     Divide the reported kinetic energy by 100 to compare against an unscaled run.
**
**  3. DUVAUT-LIONS VISCOSITY IS ZERO, down from 1e-6. It is a relaxation time. At an implicit
**     increment of ~5e-3 s, dT/eta ~ 5000 and it regularises usefully; at the explicit increment of
**     ~3.4e-7 s almost nothing relaxes per increment, and a non-zero value makes the concrete
**     respond very nearly ELASTICALLY -- a run that completes, looks plausible, and shows no
**     softening at all. Explicit integration needs no Newton to stabilise, so zero is right.
**
** 22 properties here, not 20: idx 19 is the density, idx 20 the nonlocal viscosity eta and idx 21
** the micro-inertia m_k -- the HYPERBOLIC nonlocal field of the production hef70 explicit runs:
** eta = 3e-4 s and m_k = eta^2/4 = 2.25e-8 s^2. With m_k > 0 the nonlocal damage field is integrated
** by central difference instead of forward Euler, so its stable increment falls off with h instead
** of h^2 (#134). Neither value depends on the mesh:
**  - m_k = eta^2/4 is the critically damped pairing of the homogeneous mode; Marmot rejects
**    m_k > eta^2/4. It depends on eta alone, and a larger m_k only raises the nonlocal limit.
**  - The mesh enters only through the nonlocal stable increment, ~ sqrt(m_k) * h / l, which does not
**    bind here: the steel governs dt.
**  - eta is a relaxation TIME: the nonlocal field lags the local one by ~eta, 1.25 % of this deck's
**    0.024 s ramp (0.03 % in production, stepLength = 1 s).
** The DENSITY is NOT the production value (x2000 at stepLength 1 s): the mass scaling goes with the
** loading time, and this deck keeps its own x100.
*material, name=gcdp, id=gcdp
****1****,  ****2****,  ****3****,  ****4****,  ****5****,  ****6****,  ****7****,  ****8****,
**E         nu          fcy         fcu         fbu         ftu         Df          Ah
30000.0,    0.2,        9.3,        28,         33,         2.2,        0.85,       0.08,
**Bh        Ch          Dh          As          epsF        lDamage     m           maxDmg
0.003,      2.0,        0.000001,   1,          0.0130,     2,          1.00,       0.99,
**drvdvMthd dTThrshHold viscosity   density     nonlocalViscosity  microInertia
0,          1e-12,      0,          3e-7,       3e-4,              2.25e-8

** STEEL: perfectly plastic von Mises, fy = 1080 MPa (high-strength anchor rod), as in production.
*material, name=VonMises, id=steelAnchor
**E       nu   fy      HLin deltaYield delta density (x100, mass scaled)
200000.0, 0.3, 1080.0, 0.0, 0.0,       0.0,  7.85e-7

*material, name=VonMises, id=steelPlate
200000.0, 0.3, 1080.0, 0.0, 0.0,       0.0,  7.85e-7

*material, name=VonMises, id=steelFastener
200000.0, 0.3, 1080.0, 0.0, 0.0,       0.0,  7.85e-7

*material, name=LinearElastic, id=mortar
5000.0, 0.2, 2.2e-7

**=================================================================================================
** ARTIFICIAL BULK VISCOSITY (explicit only), as in production
**=================================================================================================
** Central difference is non-dissipative; without damping the reaction rings from the first element
** that cracks through. b1 = 0.06 (linear) and b2 = 1.2 (quadratic, compression only) are the
** Abaqus/Explicit defaults, on every element set. The viscous stress is added to the integrated
** stress only, never stored, and does not touch the constitutive law.
**
** On the concrete the linear term is degraded with damage as (c/c_0)^2.0: inside a cracked element it
** is a crack-opening resistance that never relaxes, and undegraded it added +27..55 % to the
** fracture energy of a GCDP bar, all of it generated inside the damaged band.
*elementProperty, elSet=concrete, propertyName=bulk viscosity
0.06, 1.2

*elementProperty, elSet=concrete, propertyName=bulk viscosity damage degradation
2.0

*elementProperty, elSet=anchor, propertyName=bulk viscosity
0.06, 1.2

*elementProperty, elSet=plate, propertyName=bulk viscosity
0.06, 1.2

*elementProperty, elSet=washer, propertyName=bulk viscosity
0.06, 1.2

*elementProperty, elSet=nut, propertyName=bulk viscosity
0.06, 1.2

*elementProperty, elSet=mortar, propertyName=bulk viscosity
0.06, 1.2
