**=================================================================================================
** MATERIALS
**=================================================================================================
**
** GCDP (concrete), Marmot native material "gcdp". Properties from
** incfiles/uelprops_GCDP_RWW_batch_E1_l4_As1.inc, reordered/renamed to match
** Marmot/modules/materials/GCDP/src/GCDP.cpp::computeStress (idx 0-18):
**
**   idx  name        Abaqus UEL name   value
**    0   E           E                 30600.0
**    1   nu          nu                0.2
**    2   fcy         fcy               9.3
**    3   fcu         fcu               32.3
**    4   fbu         fbu               36.0
**    5   ftu         ftu               2.7
**    6   Df          Df                0.85
**    7   Ah          Ah                0.08
**    8   Bh          Bh                0.003
**    9   Ch          Ch                2.0
**   10   Dh          Dh                0.000001
**   11   As          As                1.0
**   12   softeningModulus   epsF       0.0037
**   13   nonLocalRadius     lDamage    4
**   14   weightingParameter m          1.00
**   15   omegaMax           maxDmg     0.995
**   16   derivativeMethod   drvdvMthd  0
**   17   dTThresholdElasticStiffness  dTThrshHold  1e-12
**   18   viscosity          viscosity  1e-5
**
** NOTE: Marmot/modules/materials/GCDP/src/GCDP.cpp's own GCDPModel::getDensity() hard-indexes
** materialProperties[18] as density, which is the SAME slot computeStress reads as "viscosity"
** above (looks like an off-by-one left over from adding the viscosity parameter -- getDensity
** documents idx 18, but computeStress's own index arithmetic lands viscosity there too). We never
** RESOLVED: fixed in the GCDP module (fix/density-property-index) -- getDensity now reads idx 19 and
** getNonlocalViscosity idx 20, leaving idx 18 to computeStress. This solver DOES call getDensity(),
** on every element, so the collision was not harmless here.
*material, name=gcdp, id=gcdp
****E         nu      fcy    fcu    fbu   ftu   Df      Ah
**30600.0,    0.2,    9.3,   32.3,  36.0, 2.7,  0.85,   0.08,
****Bh        Ch      Dh        As    softmod   l   m      omegaMax
**0.003,      2.0,    0.000001, 1.0,  0.0037,   4,  1.00,  0.995,
****diffmeth  dTThreshold viscosity
**0,          1e-12,      1e-5
***UEL Property, Elset=concrete-1.CONCRETE
** First are the material (GCDP) properties, then the element properties (none in this case)
****1****,  ****2****,  ****3****,  ****4****,  ****5****,  ****6****,  ****7****,  ****8****,
**E         nu          fcy         fcu         fbu         ftu         Df          Ah
30000.0,    0.2,        9.3,        28,         33,         2.2,      0.85,       0.08,
**Bh        Ch          Dh          As          epsF,       lDamage     m           maxDmg
0.003,      2.0,        0.000001,   1,         0.0130,     2,          1.00,       0.99,
**drvdvMthd dTThrshHold viscosity   density     nonlocalViscosity
0,          1e-12,      0,          3e-7,       1e-5

*material, name=LinearElastic, id=steelAnchor
**E       nu   density (x100, mass scaled)
200000.0, 0.3, 7.85e-7

*material, name=LinearElastic, id=steelPlate
**E       nu   density (x100, mass scaled)
200000.0, 0.3, 7.85e-7

** washer + nut fixture: kept as its own material id (matching the source model's intent of
** varying fastener stiffness independently), currently identical to the other steel parts
*material, name=LinearElastic, id=steelFastener
**E       nu   density (x100, mass scaled)
200000.0, 0.3, 7.85e-7

** mortar/grout equivalent solid -- see simplification note 3 above
*material, name=LinearElastic, id=mortar
**E     nu   density (x100, mass scaled)
5000.0, 0.2, 2.2e-7
