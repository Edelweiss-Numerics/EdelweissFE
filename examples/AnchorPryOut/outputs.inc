**=================================================================================================
** FIELD OUTPUTS
**=================================================================================================
*fieldOutput
>>perElement, name=alphaP, elSet=concrete, result=alphaP, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
>>perElement, name=damage, elSet=concrete, result=omega, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
** The AMR marker's own input, deliberately a separate declaration from 'alphaP' above: changing
** what the marker sees must not silently change what is exported, nor the other way round. The
** quantity is the same -- GCDP's hardening variable, maxed over the element's quadrature points.
>>perElement, name=alphaPForAMR, elSet=concrete, result=alphaP, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
>>perNode, name=dispConcrete, elSet=concrete, field=displacement, result=U
>>perNode, name=dispAnchor, elSet=anchor, field=displacement, result=U
>>perNode, name=dispPlate, elSet=plate, field=displacement, result=U
>>perNode, name=dispMortar, elSet=mortar, field=displacement, result=U
>>perNode, name=dispNut, elSet=nut, field=displacement, result=U
>>perNode, name=dispWasher, elSet=washer, field=displacement, result=U
>>perNode, name=nonlocalDamage, elSet=concrete, field=nonlocal damage, result=U
>>perNode, name=RF_loading, nSet=plate_left_loading, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=RF_loading
>>perNode, name=U_loading, nSet=plate_left_loading, field=displacement, result=U, f(x)='np.mean(x[:,0])', saveHistory=True, export=U_loading
** The peak kinetic energy has to be compared against the work the loading actually did, and the
** solver's internal-energy figure cannot supply that: GCDP never populates a strain energy, so the
** internal/kinetic split it prints is identically zero on the internal side. Integrating RF_loading
** against U_loading (trapezoidal, from these two CSVs) needs no cooperation from the material and is
** the criterion to use when choosing stepLength.
>>perElement, name=maxDamage, elSet=concrete, result=omega, quadraturePoint=0:8, f(x)='np.max(x)', saveHistory=True, export=maxDamage

*output, type=ensight, name=esExport
>>perNode, fieldOutput=dispConcrete
>>perNode, fieldOutput=dispAnchor
>>perNode, fieldOutput=dispPlate
>>perNode, fieldOutput=dispMortar
>>perNode, fieldOutput=dispNut
>>perNode, fieldOutput=dispWasher
>>perNode, fieldOutput=nonlocalDamage
>>perElement, fieldOutput=damage
>>perElement, fieldOutput=alphaP
>>configuration, overwrite=yes

*output, type=monitor, name=monitor
fieldOutput=RF_loading
fieldOutput=U_loading

** Checkpoints, so an interrupted run resumes instead of restarting from zero. A ring buffer of
** 3 bounds the disk cost; the interval is a compromise between how much work an interruption
** throws away and how often the full model state is serialised.
*output, type=restart, name=restart
writeInterval=2, baseName=restart, numberOfFilesToKeep=3
