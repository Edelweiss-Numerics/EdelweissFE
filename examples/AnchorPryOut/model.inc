**=================================================================================================
** MODEL  --  mesh, sections, facet surfaces and ties. Shared by every variant.
**=================================================================================================
** Verified identical between the implicit and explicit decks this file was factored out of:
** the mesh includes, all five sections, all fourteen facet-surface generators and all five
** tie constraints matched line for line. Only the materials, the solver, the field outputs and
** the steps ever differed, and those are the four includes a variant actually chooses.
**
**=================================================================================================
** TRANSLATED FROM Abaqus/Standard AnchorPryOut.inp -- the decisions that are not free parameters
**=================================================================================================
** 1. Concrete: Abaqus UEL U004 (C3D20R host, GCDP material) -> Marmot element GC3D20R (gradient-
**    enhanced hex20, reduced integration) with Marmot's native "gcdp" material. Property order
**    verified against Marmot/modules/materials/GCDP/src/GCDP.cpp (19 properties; the Abaqus UEL's
**    20th/21st properties, viscosity and density, collide at material property index 18 in the
**    current Marmot GCDP source -- see the comment above the material block below).
**
** 2. Steel (anchor/plate/washer/nut): Abaqus C3D8I (incompatible-modes hex8) -> Marmot C3D8
**    (standard, fully-integrated hex8). Marmot does not currently register an incompatible-modes/
**    enhanced-assumed-strain hex8 under any element name. This is slightly stiffer in bending than
**    the source model; not expected to matter here since these parts stay close to linear-elastic
**    and are not the focus of the analysis, but worth knowing if plate bending becomes significant.
**
** 4. Node numbering: EdelweissFE has no Abaqus Part/Instance/Assembly concept -- one flat node and
**    element label space. Concrete (mesh/concrete.inp, nodes 1-58569) and steel (mesh/steel.inp,
**    nodes 4489-19732) node numbers collide, so mesh/steel_edelweissfe.inp is a copy of
**    mesh/steel.inp with every node label (and every reference to one, in *ELEMENT connectivity
**    and *NSET lists) offset by +100000. Element labels do NOT collide (concrete: 4004-15575,
**    steel: 1-4003) and were left untouched. *ELSET/*SURFACE blocks reference element labels only,
**    also untouched. The *NSET "z_symm" is defined in both parts and was renamed to
**    "z_symm_concrete"/"z_symm_steel" to avoid one silently overwriting the other. See
**    mesh/concrete_edelweissfe.inp / mesh/steel_edelweissfe.inp for the exact (small, mechanical)
**    diffs against the original Abaqus mesh files -- geometry/connectivity/sets are otherwise
**    byte-for-byte identical.
**
** 6. The *SURFACE blocks below are the original Abaqus S1..S6 face-numbered elsets, imported
**    unmodified and fed straight into "*modelGenerator, generator=surfaceElementGenerator".
**    EdelweissFE's own face-numbering convention for this generator is documented as NOT being
**    the general Abaqus S1..S6 convention (see edelweissfe/generators/surfaceelementgenerator.py)
**    -- checked empirically (facet span/shape sanity check + closest-point-distance profiling
**    for every tie pair, see point 7): all 5 generated facet surfaces are geometrically sane
**    (small, local, correctly-shaped facets; matching bounding boxes / radii between tie pairs),
**    so this coincidence holds for this mesh's Marmot-compatible node ordering.
**
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
