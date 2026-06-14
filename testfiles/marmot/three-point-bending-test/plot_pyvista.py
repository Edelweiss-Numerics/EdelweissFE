import argparse
import numpy as np
import pyvista as pv

# ==========================================
# COMMAND LINE ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(
    description="Compare two EnSight Gold FEA results for RILEM TPB with a joint load-displacement plot."
)
parser.add_argument("file1", type=str, help="Path to the FIRST EnSight .case file")
parser.add_argument("file2", type=str, help="Path to the SECOND EnSight .case file")
parser.add_argument(
    "-w",
    "--warp",
    type=float,
    default=1.0,
    help="Displacement magnification factor (default: 1.0)",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="tpb_comparison.mp4",
    help="Output MP4 filename (default: tpb_comparison.mp4)",
)
args = parser.parse_args()

# Configuration
file1 = args.file1
file2 = args.file2
warp_factor = args.warp
output_file = args.output

# 3D Field Rendering Variables
disp_var = "Displacement"

# 2D Chart Extraction Variables
chart_disp_var = "DispTop"
force_var = "ReactionTop"

disp_set_name = "NSET_top_center_load_line"
force_set_name = "NSET_top_center_load_line" 

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def initialize_reader(filepath):
    """Initializes the PyVista reader and auto-selects the transient time set."""
    reader = pv.get_reader(filepath)
    try:
        best_set = 0
        max_steps = 0
        for i in range(5): 
            try:
                reader.set_active_time_set(i)
                current_steps = len(reader.time_values)
                if current_steps > max_steps:
                    max_steps = current_steps
                    best_set = i
            except Exception:
                break
                
        reader.set_active_time_set(best_set)
        print(f"Loaded '{filepath}': Selected Time Set {best_set} with {max_steps} timesteps.")
    except AttributeError:
        pass
        
    return reader

def find_block_by_substring(multiblock, substring):
    """Recursively search for a block containing a specific substring name."""
    for i in range(multiblock.n_blocks):
        name = multiblock.get_block_name(i)
        if name and substring in name:
            return multiblock[i]
        if isinstance(multiblock[i], pv.MultiBlock):
            res = find_block_by_substring(multiblock[i], substring)
            if res is not None:
                return res
    return None

def extract_block(multiblock, name):
    """Recursively find a block by its exact name."""
    for i in range(multiblock.n_blocks):
        if multiblock.get_block_name(i) == name:
            return multiblock[i]
        if isinstance(multiblock[i], pv.MultiBlock):
            res = extract_block(multiblock[i], name)
            if res is not None:
                return res
    raise ValueError(f"Block '{name}' not found in the dataset.")

def pre_calculate_curves(reader):
    """Loops through all times to extract force/displacement histories for a single model."""
    times = np.array(reader.time_values)
    displacements = []
    forces = []

    for t in times:
        reader.set_active_time_value(t)
        data = reader.read()

        # Extract Force
        force_block = find_block_by_substring(data, force_set_name)
        if force_block is not None and force_var in force_block.point_data:
            f_data = force_block.point_data[force_var]
            # Safely handle 3D vector vs 1D scalar arrays
            if f_data.ndim > 1:
                sum_force = np.sum(np.abs(f_data[:, 1]))
            else:
                sum_force = np.sum(np.abs(f_data))
        else:
            sum_force = 0.0

        # Extract Displacement
        disp_block = find_block_by_substring(data, disp_set_name)
        if disp_block is not None and chart_disp_var in disp_block.point_data:
            d_data = disp_block.point_data[chart_disp_var]
            # Safely handle 3D vector vs 1D scalar arrays
            if d_data.ndim > 1:
                avg_disp = np.mean(np.abs(d_data[:, 1]))
            else:
                avg_disp = np.mean(np.abs(d_data))
        else:
            avg_disp = 0.0

        forces.append(sum_force)
        displacements.append(avg_disp)

    return times, np.array(displacements), np.array(forces)

def get_interpolated_state(reader, times, disp_hist, force_hist, t_target):
    """Calculates the interpolated mesh and tracking variables for a specific time."""
    t = min(t_target, times[-1])
    
    idx_after = np.searchsorted(times, t)
    idx_before = max(0, idx_after - 1)
    if idx_after >= len(times):
        idx_after = len(times) - 1

    t_before = times[idx_before]
    t_after = times[idx_after]
    weight = (t - t_before) / (t_after - t_before) if t_after != t_before else 0.0

    reader.set_active_time_value(t_before)
    data_before = reader.read()
    reader.set_active_time_value(t_after)
    data_after = reader.read()

    mesh_before = extract_block(data_before, "all")
    mesh_after = extract_block(data_after, "all")

    current_mesh = mesh_before.copy(deep=True)
    interp_disp = mesh_before.point_data[disp_var] + weight * (
        mesh_after.point_data[disp_var] - mesh_before.point_data[disp_var]
    )
    current_mesh.point_data[disp_var] = interp_disp
    current_mesh.points += interp_disp * warp_factor

    current_disp = disp_hist[idx_before] + weight * (disp_hist[idx_after] - disp_hist[idx_before])
    current_force = force_hist[idx_before] + weight * (force_hist[idx_after] - force_hist[idx_before])

    history_mask = times <= t
    current_disp_line = list(disp_hist[history_mask]) + [current_disp]
    current_force_line = list(force_hist[history_mask]) + [current_force]

    return current_mesh, current_disp_line, current_force_line

# ==========================================
# INITIALIZATION & PRE-PROCESSING
# ==========================================
print("Initializing Model 1...")
reader1 = initialize_reader(file1)
times1, plot_disp1, plot_forces1 = pre_calculate_curves(reader1)

print("Initializing Model 2...")
reader2 = initialize_reader(file2)
times2, plot_disp2, plot_forces2 = pre_calculate_curves(reader2)

# Global plot limits
max_disp = max(np.max(plot_disp1), np.max(plot_disp2)) * 1.1
min_force = 0.0  
max_force = max(np.max(plot_forces1), np.max(plot_forces2)) * 1.1

if max_disp <= 1e-9 and max_force <= 1e-9:
    print("WARNING: Data arrays are nearly zero for both models. Please check node set and variable names.")

global_min_time = min(times1.min(), times2.min())
global_max_time = max(times1.max(), times2.max())

# ==========================================
# ANIMATION & RENDERING
# ==========================================
plotter = pv.Plotter(shape=(1, 2), window_size=[1920, 960])
plotter.open_movie(output_file, framerate=15)

sargs = dict(
    title_font_size=24,
    label_font_size=24,
    vertical=False,
    position_x=0.1,
    position_y=0.05,
    height=0.08,
    width=0.8,
)

plotter.subplot(0, 1)
chart = pv.Chart2D(size=(0.4, 0.35), loc=(0.55, 0.05))
chart.title = "TPB Load-Displacement"
chart.x_label = "Deflection (Y)"
chart.y_label = "Reaction Force (Y)"
chart.x_range = [0, max_disp if max_disp > 0 else 1]
chart.y_range = [0, max_force if max_force > 0 else 1]
plotter.add_chart(chart)

line1 = chart.line([0], [0], color="red", width=3, label="Model 1")
line2 = chart.line([0], [0], color="blue", width=3, label="Model 2")

first_frame = True
eval_times = np.linspace(global_min_time, global_max_time, 120)

for t in eval_times:
    
    mesh1, disp_line1, force_line1 = get_interpolated_state(reader1, times1, plot_disp1, plot_forces1, t)
    mesh2, disp_line2, force_line2 = get_interpolated_state(reader2, times2, plot_disp2, plot_forces2, t)

    line1.update(disp_line1, force_line1)
    line2.update(disp_line2, force_line2)

    plotter.subplot(0, 0)
    plotter.add_mesh(
        mesh1,
        scalars=disp_var,
        cmap="coolwarm",
        show_edges=True,
        edge_color="gray",
        scalar_bar_args=sargs,
        reset_camera=False,
        name="model_1",
    )
    plotter.add_text("Model 1", font_size=14, position="upper_left", name="label_1")

    plotter.subplot(0, 1)
    plotter.add_mesh(
        mesh2,
        scalars=disp_var,
        cmap="coolwarm",
        show_edges=True,
        edge_color="gray",
        scalar_bar_args=sargs,
        reset_camera=False,
        name="model_2",
    )
    plotter.add_text("Model 2", font_size=14, position="upper_left", name="label_2")

    if first_frame:
        plotter.link_views()
        for i in [0, 1]:
            plotter.subplot(0, i)
            plotter.view_xy()
            plotter.reset_camera()
            
        plotter.show(auto_close=False, interactive_update=True)
        first_frame = False
    else:
        plotter.update()

    plotter.write_frame()

# ==========================================
# CLEANUP
# ==========================================
print(f"Rendering complete! Close the preview window to finish saving your video to: '{output_file}'")
plotter.show()
