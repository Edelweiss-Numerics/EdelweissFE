"""Hanging-node search: every grid cell around a coarse element must be searched for finer neighbours.

:meth:`AdaptiveMesh.classify_hanging` only scans coarse elements that have a strictly finer
neighbour. That test first checks cheaply whether ANY grid cell around the element holds a finer
element, and then searches those cells for one that actually touches it. Both passes must visit the
same cells: if the first pass consumes part of the cell sequence, the cell in which it found the
finer element is never searched again, and a coarse element whose finer neighbours all sit in that
one cell is wrongly skipped -- its hanging nodes get no constraint.

On a mesh with exactly shared nodes this cannot happen: a finer neighbour is at least one grid cell
long along its longest axis, so it always reaches into a second cell that is still searched. It
takes a neighbour that ends within the geometric tolerance *before* a grid line -- e.g. coordinates
carrying round-off -- to confine it to a single cell. The fixture builds exactly that: an element B
that ends 5e-9 (less than the 1e-8 classification tolerance) before the face x = 0 of a coarse
element A and is only 0.99 wide in y and z, so all four children of B that touch A lie in one cell
of A's search window.
"""

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh, _grid_cells_for_box

GAP = 5e-9
COARSE_A = hex20_box_coords(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
REFINED_B = hex20_box_coords(-2.0 - GAP, -GAP, 0.0, 0.99, 0.0, 0.99)


def _meshWithRefinedNeighbour():
    mesh = AdaptiveMesh(splitFactor=2, topology=Hex20Topology())
    coarse = mesh.add_root(COARSE_A, 0)
    mesh.refine(mesh.add_root(REFINED_B, 0))
    return mesh, coarse


def test_the_fixture_puts_all_touching_children_into_one_cell():
    """Guard the fixture: if the children touching A spread over several grid cells, the cell the
    cheap test stopped at is not the only one holding them, and the test below proves nothing."""
    mesh, coarse = _meshWithRefinedNeighbour()
    active = mesh.active()
    cellSize = mesh._cellSize(active)
    searchWindowOfA = set(_grid_cells_for_box(*mesh.box(coarse), cellSize))
    touchingChildren = [eid for eid in active if eid != coarse and mesh.box(eid)[1][0] > -1.0]
    assert len(touchingChildren) == 4
    cellsOfTouchingChildren = set()
    for eid in touchingChildren:
        cellsOfTouchingChildren |= set(_grid_cells_for_box(*mesh.box(eid), cellSize, pad=0)) & searchWindowOfA
    assert len(cellsOfTouchingChildren) == 1


def test_a_coarse_element_whose_finer_neighbours_share_one_cell_hosts_their_hanging_nodes():
    mesh, coarse = _meshWithRefinedNeighbour()
    nodesOfA = set(mesh.elements[coarse]["conn"])
    hangingOnA = [h for h in mesh.classify_hanging() if set(h["masters"]) <= nodesOfA]
    assert hangingOnA, "the children of B hang on the face of A, but A was never scanned for hanging nodes"
