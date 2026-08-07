import cubit


def build_rilem_tpb():
    # ---------------------------------------------------------
    # 1. Parameter Definitions (Standard RILEM Proportions)
    # ---------------------------------------------------------
    L = 440.0  # Total length of the beam
    S = 400.0  # Span length (distance between supports)
    D = 100.0  # Height of the beam
    B = 10.0  # Thickness of the beam

    a = 25.0  # Notch depth (typically D/2)
    w_n = 10.0  # Notch width

    # Mesh definitions
    mesh_size = 10.0  # General element size
    thickness_intervals = 1  # Number of elements through the thickness

    # Calculated X-coordinates for boundary conditions
    x_supp_left = (L - S) / 2.0
    x_supp_right = (L + S) / 2.0
    x_center = L / 2.0

    # ---------------------------------------------------------
    # 2. Geometry Creation & Boolean Operations
    # ---------------------------------------------------------
    cubit.cmd("reset")

    # Create the main beam brick and position it in the positive quadrant
    cubit.cmd(f"create brick x {L} y {D} z {B}")
    cubit.cmd(f"move volume 1 x {L / 2} y {D / 2} z {B / 2}")

    # Create the notch brick (make it slightly longer in Z to ensure a clean cut)
    cubit.cmd(f"create brick x {w_n} y {a} z {B + 2.0}")
    cubit.cmd(f"move volume 2 x {L / 2} y {a / 2} z {B / 2}")

    # Subtract the notch from the main beam
    cubit.cmd("subtract volume 2 from volume 1")

    # ---------------------------------------------------------
    # 3. Strategic Webcuts (For mapped meshing and exact BC nodes)
    # ---------------------------------------------------------
    # Slicing the volume at critical locations guarantees that nodes will fall
    # exactly on our support and loading lines, and forces the volumes into pure bricks.

    cubit.cmd(f"webcut volume all with plane xplane offset {x_supp_left}")
    cubit.cmd(f"webcut volume all with plane xplane offset {x_center - w_n / 2.0}")
    cubit.cmd(f"webcut volume all with plane xplane offset {x_center}")
    cubit.cmd(f"webcut volume all with plane xplane offset {x_center + w_n / 2.0}")
    cubit.cmd(f"webcut volume all with plane xplane offset {x_supp_right}")

    # Tie the discrete volumes back together so the mesh is continuous
    cubit.cmd("imprint volume all")
    cubit.cmd("merge volume all")

    # ---------------------------------------------------------
    # 4. Meshing
    # ---------------------------------------------------------
    # Apply global sizing
    cubit.cmd(f"volume all size {mesh_size}")

    # Override the thickness interval to force a specific number of element rows
    # Selects all curves oriented along the Z-axis (centroids between Z=0 and Z=B)
    cubit.cmd(f"curve with z_coord > {B * 0.1} and z_coord < {B * 0.9} interval {thickness_intervals}")

    # Mesh the volumes using the mapped scheme
    cubit.cmd("volume all scheme map")
    cubit.cmd("mesh volume all")

    # ---------------------------------------------------------
    # 5. Element Blocks
    # ---------------------------------------------------------
    cubit.cmd("block 1 volume all")
    cubit.cmd("block 1 name 'beam_elements'")
    cubit.cmd("block 1 element type hex8")

    # ---------------------------------------------------------
    # 6. Boundary Conditions (Node Sets)
    # ---------------------------------------------------------
    tol = 0.001  # Tolerance for geometric bounding boxes

    # Supports (Bottom lines along the Z-axis)
    cubit.cmd(
        f"nodeset 1 add node with x_coord > {x_supp_left - tol} and x_coord < {x_supp_left + tol} and y_coord < {tol}"
    )
    cubit.cmd("nodeset 1 name 'left_support'")

    cubit.cmd(
        f"nodeset 2 add node with x_coord > {x_supp_right - tol} and x_coord < {x_supp_right + tol} and y_coord < {tol}"
    )
    cubit.cmd("nodeset 2 name 'right_support'")

    # Loading Point (Top center line along the Z-axis)
    cubit.cmd(
        f"nodeset 3 add node with x_coord > {x_center - 5 - tol} and x_coord < {x_center + 5 + tol} and y_coord > {D - tol}"
    )
    cubit.cmd("nodeset 3 name 'top_center_load_line'")

    # Front and Back faces (To lock Z displacements if assuming plane strain)
    cubit.cmd(f"nodeset 4 add node in surface with z_coord > {B - tol}")
    cubit.cmd("nodeset 4 name 'front_nodes'")

    cubit.cmd(f"nodeset 5 add node in surface with z_coord < {tol}")
    cubit.cmd("nodeset 5 name 'back_nodes'")

    # ---------------------------------------------------------
    # 7. Abaqus Export
    # ---------------------------------------------------------
    cubit.cmd('export abaqus "rilem_tpb.inp" partial overwrite')


# Execute the main function
build_rilem_tpb()
