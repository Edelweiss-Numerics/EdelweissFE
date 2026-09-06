**=================================================================================================
** MODEL  --  mesh, sections, facet surfaces and ties. Shared by every variant.
**=================================================================================================
** Verified identical between the implicit and explicit decks this file was factored out of:
** the mesh includes, all five sections, all fourteen facet-surface generators and all five
** tie constraints matched line for line. Only the materials, the solver, the field outputs and
** the steps ever differed, and those are the four includes a variant actually chooses.
**
**=================================================================================================
** MESH
**=================================================================================================
*include, input=./mesh/concrete_edelweissfe.inp
*include, input=./mesh/steel_edelweissfe.inp

** The z-symmetry node sets come from the MESH, not from a generated side file. The Cubit journal
** declares them as 'node in volume all expand with z_coord = 0', and that is complete: of the
** 56 881 nodes, 5 259 lie on the z = 0 plane and all 5 259 are members, with none declared that are
** off it. An earlier revision of this mesh was short by 516 nodes and was patched by including a
** geometrically-derived 'z_symm_all' set alongside; that mesh has since been regenerated, the
** patch became a byte-for-byte duplicate of what the mesh already declares, and it is gone.

**=================================================================================================
** SECTIONS
**=================================================================================================
*section, name=concreteSection, material=gcdp, type=solid
concrete

*section, name=anchorSection, material=steelAnchor, type=solid
anchor

*section, name=plateSection, material=steelPlate, type=solid
plate

*section, name=fastenerSection, material=steelFastener, type=solid
washer
nut

*section, name=mortarSection, material=mortar, type=solid
mortar

**=================================================================================================
** CONTACT / TIE FACET SURFACES
**=================================================================================================
** Generates flat contact-facet elements from the imported Abaqus *SURFACE blocks, for use by the
** tie and penalty-contact constraints below. triangulation=midside used uniformly: mandatory for
** the concrete's hexa20 faces, harmless (falls back to the linear corner table) for the steel/
** mortar hexa8 faces.

*modelGenerator, generator=surfaceElementGenerator, name=genConcreteTop, executeAfterManualGeneration=True
surface = concrete_top
name    = concrete_top
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genConcreteToMortar, executeAfterManualGeneration=True
surface = concrete_to_mortar
name    = concrete_to_mortar
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateBottom, executeAfterManualGeneration=True
surface = plate_bottom
name    = plate_bottom
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateTop, executeAfterManualGeneration=True
surface = plate_top
name    = plate_top
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genMortarToConcrete, executeAfterManualGeneration=True
surface = mortar_to_concrete
name    = mortar_to_concrete
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genMortarToAnchor, executeAfterManualGeneration=True
surface = mortar_to_anchor
name    = mortar_to_anchor
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToMortar, executeAfterManualGeneration=True
surface = anchor_to_mortar
name    = anchor_to_mortar
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToPlate, executeAfterManualGeneration=True
surface = anchor_to_plate
name    = anchor_to_plate
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateToAnchor, executeAfterManualGeneration=True
surface = plate_to_anchor
name    = plate_to_anchor
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genWasherBottom, executeAfterManualGeneration=True
surface = washer_bottom
name    = washer_bottom
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genWasherToNut, executeAfterManualGeneration=True
surface = washer_to_nut
name    = washer_to_nut
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genNutToWasher, executeAfterManualGeneration=True
surface = nut_to_washer
name    = nut_to_washer
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genNutToAnchor, executeAfterManualGeneration=True
surface = nut_to_anchor
name    = nut_to_anchor
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToNut, executeAfterManualGeneration=True
surface = anchor_to_nut
name    = anchor_to_nut
triangulation = midside

**=================================================================================================
** TIE CONSTRAINTS (Abaqus *TIE, TYPE=SURFACE TO SURFACE -> EdelweissFE type=tie)
**=================================================================================================
** first-listed surface is the slave (matches the source model's own convention)

*constraint, name=tieNutToAnchor, type=tie
slaveSurface=anchor_to_nut_facets, masterSurface=nut_to_anchor_facets

** POSITION TOLERANCE=0.6 in the source model: covers the DIN 125 clearance bore gap between the
** washer's inner bore and the nut's bottom radius (see AnchorPryOut.inp comment)
*constraint, name=tieNutToWasher, type=tie
slaveSurface=nut_to_washer_facets, masterSurface=washer_to_nut_facets, positionTolerance=0.6

*constraint, name=tieWasherToPlate, type=tie
slaveSurface=washer_bottom_facets, masterSurface=plate_top_facets

*constraint, name=tieAnchorToMortar, type=tie
slaveSurface=mortar_to_anchor_facets, masterSurface=anchor_to_mortar_facets

** Full bond length: concrete_to_mortar (the concrete borehole wall) and mortar_to_concrete (the
** mortar solid's outer face) both now span the same 80 mm embedment depth (see header point 7 --
** the borehole used to erroneously extend the full 240 mm slab height before a Cubit geometry
** fix). All 373 slave nodes tie (0 untied); no explicit positionTolerance needed, the default
** (positionToleranceFactor=0.25 times mortar_to_concrete's mean facet size) covers the sub-mm
** mesh-discretization gap comfortably on its own.
*constraint, name=tieMortarToConcrete, type=tie
slaveSurface=concrete_to_mortar_facets, masterSurface=mortar_to_concrete_facets
