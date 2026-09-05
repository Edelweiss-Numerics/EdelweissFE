**=================================================================================================
** MATERIALS -- implicit
**=================================================================================================
** Concrete: Marmot's GCDP, a gradient-enhanced damaged-plasticity model. The gradient enhancement
** is what makes the softening response mesh-objective; lDamage (idx 13) is its internal length.
**
** 20 properties: idx 18 is the Duvaut-Lions viscosity, idx 19 the density. An implicit static solve
** builds no mass matrix, so the steel densities are omitted here and the explicit variant declares
** its own (see materials_explicit.inc).
**
** The Duvaut-Lions viscosity is a relaxation time and acts as mild numerical regularisation at an
** implicit increment of ~5e-3. It is set to ZERO in the explicit variant -- see there for why.
*material, name=gcdp, id=gcdp
****1****,  ****2****,  ****3****,  ****4****,  ****5****,  ****6****,  ****7****,  ****8****,
**E         nu          fcy         fcu         fbu         ftu         Df          Ah
30000.0,    0.2,        9.3,        28,         33,         2.2,        0.85,       0.08,
**Bh        Ch          Dh          As          epsF        lDamage     m           maxDmg
0.003,      2.0,        0.000001,   1,          0.0130,     2,          1.00,       0.99,
**drvdvMthd dTThrshHold viscosity   density
0,          1e-12,      1e-6,       3e-9,

*material, name=LinearElastic, id=steelAnchor
200000.0, 0.3

*material, name=LinearElastic, id=steelPlate
200000.0, 0.3

*material, name=LinearElastic, id=steelFastener
200000.0, 0.3

** Mortar/grout as an equivalent linear-elastic solid. The source Abaqus model used COH3D8 cohesive
** elements whose stack direction is circumferential, so their bulk stiffness carries essentially
** nothing in the radial/axial load path -- that is carried by the two ties either side.
*material, name=LinearElastic, id=mortar
5000.0, 0.2
