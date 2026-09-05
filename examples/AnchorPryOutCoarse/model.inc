**=================================================================================================
** MESH, SECTIONS, INTERFACE SURFACES AND TIES  --  shared by all four variants
**=================================================================================================
** Nothing here depends on the solver or on the contact formulation, so all four decks include this
** file unchanged.
**
** THE MESH is a deliberately coarse variant of the production anchor pry-out model: an M20 bonded
** anchor at h_ef = 80 mm in a 500 x 500 x 240 mm concrete slab, loaded through a fixture plate.
** Half the specimen is modelled, with a symmetry plane at z = 0.
**
**     concrete   1634 GC3D20R  (gradient-enhanced hex20, Marmot)
**     steel        486 C3D8    (anchor, plate, washer, nut, and the mortar bond layer)
**     dof       ~29 600
**
** Regenerate it with Pryout_anchorwashernut/mesh/Pryout_elastic_anchorwashernut.py at
** PRYOUT_COARSEN=2.0 PRYOUT_SHAFT_REFINE=1.0, then convert with
** examples/AnchorPryOut/make_edelweissfe_mesh.py. PRYOUT_COARSEN multiplies every mesh size; at 1.0
** the script reproduces the production mesh exactly. PRYOUT_SHAFT_REFINE=1.0 drops the half-size
** refinement of the shaft above the concrete surface, which is what otherwise puts a 1.25 mm element
** in the model and governs the explicit stable time increment; the smallest steel edge is then the
** 3.8 mm washer thickness, a fixed geometric feature no further coarsening can change.
**
** Check a regenerated mesh with examples/AnchorPryOut/check_mesh.py before building on it.
*include, input=./mesh/concrete.inp
*include, input=./mesh/steel.inp

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
** INTERFACE SURFACES
**=================================================================================================
** Flat contact-facet elements generated from the mesh's Abaqus *SURFACE blocks, consumed by the
** ties and contacts below. triangulation=midside throughout: mandatory for the concrete's hexa20
** faces, and harmless on the steel/mortar hexa8 faces, where it falls back to the linear table.
*modelGenerator, generator=surfaceElementGenerator, name=genConcreteTop, executeAfterManualGeneration=True
surface=concrete_top
name=concrete_top
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genConcreteToMortar, executeAfterManualGeneration=True
surface=concrete_to_mortar
name=concrete_to_mortar
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateBottom, executeAfterManualGeneration=True
surface=plate_bottom
name=plate_bottom
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateTop, executeAfterManualGeneration=True
surface=plate_top
name=plate_top
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genMortarToConcrete, executeAfterManualGeneration=True
surface=mortar_to_concrete
name=mortar_to_concrete
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genMortarToAnchor, executeAfterManualGeneration=True
surface=mortar_to_anchor
name=mortar_to_anchor
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToMortar, executeAfterManualGeneration=True
surface=anchor_to_mortar
name=anchor_to_mortar
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToPlate, executeAfterManualGeneration=True
surface=anchor_to_plate
name=anchor_to_plate
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genPlateToAnchor, executeAfterManualGeneration=True
surface=plate_to_anchor
name=plate_to_anchor
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genWasherBottom, executeAfterManualGeneration=True
surface=washer_bottom
name=washer_bottom
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genWasherToNut, executeAfterManualGeneration=True
surface=washer_to_nut
name=washer_to_nut
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genNutToWasher, executeAfterManualGeneration=True
surface=nut_to_washer
name=nut_to_washer
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genNutToAnchor, executeAfterManualGeneration=True
surface=nut_to_anchor
name=nut_to_anchor
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=genAnchorToNut, executeAfterManualGeneration=True
surface=anchor_to_nut
name=anchor_to_nut
triangulation=midside

**=================================================================================================
** TIES
**=================================================================================================
** The first-listed surface is the slave. The fastener stack and the bonded length are tied; only
** the two genuinely unilateral interfaces are contact (see contact_*.inc).
*constraint, name=tieNutToAnchor, type=tie
slaveSurface=anchor_to_nut_facets, masterSurface=nut_to_anchor_facets

** positionTolerance covers the DIN 125 clearance between the washer bore and the nut radius.
*constraint, name=tieNutToWasher, type=tie
slaveSurface=nut_to_washer_facets, masterSurface=washer_to_nut_facets, positionTolerance=0.6

*constraint, name=tieWasherToPlate, type=tie
slaveSurface=washer_bottom_facets, masterSurface=plate_top_facets

*constraint, name=tieAnchorToMortar, type=tie
slaveSurface=mortar_to_anchor_facets, masterSurface=anchor_to_mortar_facets

** The bonded length: borehole wall to mortar, over the full 80 mm embedment. This is the interface
** that h-adaptivity refines, so it is also the one that exercises a tie under live refinement.
*constraint, name=tieMortarToConcrete, type=tie
slaveSurface=concrete_to_mortar_facets, masterSurface=mortar_to_concrete_facets
