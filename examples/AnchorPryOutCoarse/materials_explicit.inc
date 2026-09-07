**=================================================================================================
** MATERIALS -- explicit
**=================================================================================================
** Same model and same parameters as materials_implicit.inc, with three differences that an explicit
** integrator forces. CONSISTENT UNITS: N, mm, MPa, tonne, s.
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
** 21 properties here, not 20: idx 19 is the density and idx 20 the nonlocal viscosity, which is the
** mass of the nonlocal damage field's forward-Euler update -- what makes an elliptic regularisation
** explicitly integrable. It also sets that field's own stability limit, roughly
** dT <~ 2*eta_nl/(1 + l^2/h^2), and NOTHING IN THE CODE CHECKS IT. Size it against the FINEST
** element the run can reach: here h = 8.3 mm after one refinement, so l^2/h^2 = 0.06 and the limit
** is ~1.9e-5 s against a dT of ~3.4e-7 -- two orders of margin.
*material, name=gcdp, id=gcdp
****1****,  ****2****,  ****3****,  ****4****,  ****5****,  ****6****,  ****7****,  ****8****,
**E         nu          fcy         fcu         fbu         ftu         Df          Ah
30000.0,    0.2,        9.3,        28,         33,         2.2,        0.85,       0.08,
**Bh        Ch          Dh          As          epsF        lDamage     m           maxDmg
0.003,      2.0,        0.000001,   1,          0.0130,     2,          1.00,       0.99,
**drvdvMthd dTThrshHold viscosity   density     nonlocalViscosity
0,          1e-12,      0,          3e-7,       1e-5

*material, name=LinearElastic, id=steelAnchor
**E       nu   density (x100, mass scaled)
200000.0, 0.3, 7.85e-7

*material, name=LinearElastic, id=steelPlate
200000.0, 0.3, 7.85e-7

*material, name=LinearElastic, id=steelFastener
200000.0, 0.3, 7.85e-7

*material, name=LinearElastic, id=mortar
5000.0, 0.2, 2.2e-7
