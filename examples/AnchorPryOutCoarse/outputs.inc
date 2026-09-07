**=================================================================================================
** FIELD OUTPUTS  --  shared by all four variants
**=================================================================================================
*fieldOutput
>>perElement, name=alphaP, elSet=concrete, result=alphaP, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
>>perElement, name=damage, elSet=concrete, result=omega, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
** The AMR marker's own input. Kept separate from 'damage' so that changing what the marker sees
** cannot silently change what is exported, and vice versa.
>>perElement, name=damageForAMR, elSet=concrete, result=omega, quadraturePoint=0:8, f(x)='np.max(x,axis=1)'
** One displacement output PER PART, not a single one over 'all'. A fieldOutput scoped to 'all'
** writes its Ensight variable onto one merged part covering every element, which leaves the six
** real parts -- the ones the geometry actually defines -- carrying no displacement field at all.
** The result opens in Paraview but the parts cannot be warped or coloured by displacement, and an
** overlapping blob sits on top of them. Scoping one output per part puts each variable on the part
** it belongs to.
>>perNode, name=dispConcrete, elSet=concrete, field=displacement, result=U
>>perNode, name=dispAnchor, elSet=anchor, field=displacement, result=U
>>perNode, name=dispPlate, elSet=plate, field=displacement, result=U
>>perNode, name=dispMortar, elSet=mortar, field=displacement, result=U
>>perNode, name=dispNut, elSet=nut, field=displacement, result=U
>>perNode, name=dispWasher, elSet=washer, field=displacement, result=U
>>perNode, name=nonlocalDamage, elSet=concrete, field=nonlocal damage, result=U
** The load-displacement history. Its integral is the external work, which is what decides whether
** the explicit run was quasi-static -- the solver's own internal-energy figure cannot supply that,
** since GCDP never populates a strain energy.
>>perNode, name=RF_loading, nSet=plate_left_loading, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=RF_loading
>>perNode, name=U_loading, nSet=plate_left_loading, field=displacement, result=U, f(x)='np.mean(x[:,0])', saveHistory=True, export=U_loading
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
