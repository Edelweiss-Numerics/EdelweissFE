import cubit

def build_cooks_membrane():
    # ---------------------------------------------------------
    # 1. Parameter Definitions
    # ---------------------------------------------------------
    L = 48.0        # Length of the membrane (X-direction)
    H1 = 44.0       # Height of the left clamped edge
    H2 = 16.0       # Height of the right shear edge
    Y_off = 44.0    # Vertical Y-offset of the right edge
    T = 1.0         # Thickness (Z-direction)
    
    Nx = 10         # Number of elements along the longitudinal edges
    Ny = 10         # Number of elements along the vertical edges

    # ---------------------------------------------------------
    # 2. Geometry Creation
    # ---------------------------------------------------------
    cubit.cmd("reset")
    
    cubit.cmd(f"create vertex 0 0 0")
    cubit.cmd(f"create vertex {L} {Y_off} 0")
    cubit.cmd(f"create vertex {L} {Y_off + H2} 0")
    cubit.cmd(f"create vertex 0 {H1} 0")
    
    cubit.cmd("create curve vertex 1 2")
    cubit.cmd("create curve vertex 2 3")
    cubit.cmd("create curve vertex 3 4")
    cubit.cmd("create curve vertex 4 1")
    
    cubit.cmd("create surface curve 1 2 3 4")           
    cubit.cmd(f"sweep surface 1 perpendicular distance {T}") 

    # ---------------------------------------------------------
    # 3. Meshing
    # ---------------------------------------------------------
    # Set the in-plane grid
    cubit.cmd(f"curve 1 3 interval {Nx}")
    cubit.cmd(f"curve 2 4 interval {Ny}")
    
    # Isolate thickness curves
    cubit.cmd(f"curve with z_coord > {T * 0.1} and z_coord < {T * 0.9} interval 1")
    
    cubit.cmd("volume 1 scheme map")
    cubit.cmd("mesh volume 1")

    # ---------------------------------------------------------
    # 4. Element Blocks
    # ---------------------------------------------------------
    cubit.cmd("block 1 volume 1")
    cubit.cmd("block 1 name 'membrane_elements'")
    cubit.cmd("block 1 element type hex8") 

    # ---------------------------------------------------------
    # 5. Boundary Conditions (Node Sets & Side Sets)
    # ---------------------------------------------------------
    # Swapped: Front is now Z = T, Back is now Z = 0
    cubit.cmd(f"nodeset 1 add node in surface with z_coord > {T - 0.001}")
    cubit.cmd("nodeset 1 name 'front_nodes'")
    
    cubit.cmd("nodeset 2 add node in surface with z_coord < 0.001")
    cubit.cmd("nodeset 2 name 'back_nodes'")
    
    # Left Clamped Nodes
    cubit.cmd("nodeset 3 add node in surface with x_coord < 0.001")
    cubit.cmd("nodeset 3 name 'left_clamped_nodes'")
    
    # Right Shear Nodes
    cubit.cmd(f"nodeset 4 add node in surface with x_coord > {L - 0.001}")
    cubit.cmd("nodeset 4 name 'right_shear_nodes'")

    # Updated: Dedicated top-right-front node set (Z-coord moved to new front)
    cubit.cmd(f"nodeset 5 add node with x_coord > {L - 0.001} and y_coord > {Y_off + H2 - 0.001} and z_coord > {T - 0.001}")
    cubit.cmd("nodeset 5 name 'top_right_front_node'")

    # Added: Side Set for right face to apply distributed shear loads in Abaqus
    cubit.cmd(f"sideset 1 add surface with x_coord > {L - 0.001}")
    cubit.cmd("sideset 1 name 'right_shear_surface'")

    # ---------------------------------------------------------
    # 6. Abaqus Export
    # ---------------------------------------------------------
    cubit.cmd('export abaqus "cooks_membrane.inp" partial overwrite')

# Execute
build_cooks_membrane()
