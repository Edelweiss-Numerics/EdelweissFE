"""Node identity across a refinement.

A node created by subdividing an element has to be *the same node* for every parent that
touches the point -- otherwise the mesh is torn there, silently: two labels at one location,
no constraint between them, and the two halves are free to separate under load.

The regression these tests pin came from identifying nodes by their rounded coordinates. On the
anchor pry-out mesh that produced 156 duplicated nodes, one of which opened by 18.9 mm and
destroyed a 380 000-increment explicit run. The mechanism is exact, not bad luck: a mesh written
with seven significant decimals has quarter-points whose ninth decimal is a 5, i.e. exactly on the
rounding tie of an eight-decimal key,

    0.75 * (-3.636364) - 0.125 * (-7.272727) = -1.8181821249999999

so the two neighbours' last-bit differences decide which way it rounds, and they disagree.
"""

import numpy as np

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh

# Two face-adjacent GC3D20R elements (7271 and 9540) of examples/AnchorPryOut, kept verbatim:
# they share all eight nodes of one face, and the shared face carries the point
# (106.9208, -1.818182125, -54.47886), which is a rounding tie.
PARENT_A = np.array(
    [
        [np.float64(138.4114), np.float64(-7.272727), np.float64(-30.53289)],
        [np.float64(139.1247), np.float64(-7.272727), np.float64(-62.79857)],
        [np.float64(97.08204), np.float64(-7.272727), np.float64(-70.53423)],
        [np.float64(114.1268), np.float64(-7.272727), np.float64(-37.08204)],
        [np.float64(138.4114), np.float64(0.0), np.float64(-30.53289)],
        [np.float64(139.1247), np.float64(0.0), np.float64(-62.79857)],
        [np.float64(97.08204), np.float64(0.0), np.float64(-70.53423)],
        [np.float64(114.1268), np.float64(0.0), np.float64(-37.08204)],
        [np.float64(138.7681), np.float64(-7.272727), np.float64(-46.66573)],
        [np.float64(118.1034), np.float64(-7.272727), np.float64(-66.6664)],
        [np.float64(106.9208), np.float64(-7.272727), np.float64(-54.47886)],
        [np.float64(126.2691), np.float64(-7.272727), np.float64(-33.80746)],
        [np.float64(138.7681), np.float64(0.0), np.float64(-46.66573)],
        [np.float64(118.1034), np.float64(0.0), np.float64(-66.6664)],
        [np.float64(106.9208), np.float64(0.0), np.float64(-54.47886)],
        [np.float64(126.2691), np.float64(0.0), np.float64(-33.80746)],
        [np.float64(138.4114), np.float64(-3.636364), np.float64(-30.53289)],
        [np.float64(139.1247), np.float64(-3.636364), np.float64(-62.79857)],
        [np.float64(97.08204), np.float64(-3.636364), np.float64(-70.53423)],
        [np.float64(114.1268), np.float64(-3.636364), np.float64(-37.08204)],
    ]
)

PARENT_B = np.array(
    [
        [np.float64(89.73347), np.float64(0.0), np.float64(-65.19518)],
        [np.float64(105.488), np.float64(0.0), np.float64(-34.27513)],
        [np.float64(105.488), np.float64(-7.272727), np.float64(-34.27513)],
        [np.float64(89.73347), np.float64(-7.272727), np.float64(-65.19518)],
        [np.float64(97.08204), np.float64(0.0), np.float64(-70.53423)],
        [np.float64(114.1268), np.float64(0.0), np.float64(-37.08204)],
        [np.float64(114.1268), np.float64(-7.272727), np.float64(-37.08204)],
        [np.float64(97.08204), np.float64(-7.272727), np.float64(-70.53423)],
        [np.float64(97.61074), np.float64(0.0), np.float64(-49.73516)],
        [np.float64(105.488), np.float64(-3.636364), np.float64(-34.27513)],
        [np.float64(97.61074), np.float64(-7.272727), np.float64(-49.73516)],
        [np.float64(89.73347), np.float64(-3.636364), np.float64(-65.19518)],
        [np.float64(106.9208), np.float64(0.0), np.float64(-54.47886)],
        [np.float64(114.1268), np.float64(-3.636364), np.float64(-37.08204)],
        [np.float64(106.9208), np.float64(-7.272727), np.float64(-54.47886)],
        [np.float64(97.08204), np.float64(-3.636364), np.float64(-70.53423)],
        [np.float64(93.40775), np.float64(0.0), np.float64(-67.86471)],
        [np.float64(109.8074), np.float64(0.0), np.float64(-35.67859)],
        [np.float64(109.8074), np.float64(-7.272727), np.float64(-35.67859)],
        [np.float64(93.40775), np.float64(-7.272727), np.float64(-67.86471)],
    ]
)

TIE_POINT = np.array([106.9208, -1.818182125, -54.47886])


def _refineBoth():
    mesh = AdaptiveMesh(splitFactor=2, topology=Hex20Topology())
    eids = [mesh.add_root(PARENT_A, 0), mesh.add_root(PARENT_B, 0)]
    for eid in eids:
        mesh.refine(eid)
    return mesh


def _duplicateGroups(mesh, decimals=9):
    seen = {}
    for label, coord in mesh.registry.coordinates.items():
        seen.setdefault(tuple(np.round(coord, decimals)), []).append(label)
    return {k: v for k, v in seen.items() if len(v) > 1}


def test_the_fixture_really_is_face_adjacent():
    """Guard the fixture: if these two stop sharing a face, the test below proves nothing."""
    a = {tuple(np.round(c, 9)) for c in PARENT_A}
    b = {tuple(np.round(c, 9)) for c in PARENT_B}
    assert len(a & b) == 8


def test_a_point_on_a_shared_face_gets_exactly_one_label():
    mesh = _refineBoth()
    duplicates = _duplicateGroups(mesh)
    assert not duplicates, "refinement minted {:d} node(s) twice, e.g. {:s}".format(
        len(duplicates), repr(next(iter(duplicates.items())))
    )


def test_the_rounding_tie_point_is_a_single_node():
    """The specific point that tore the pry-out mesh open."""
    mesh = _refineBoth()
    hits = [
        label
        for label, coord in mesh.registry.coordinates.items()
        if np.allclose(coord, TIE_POINT, rtol=0.0, atol=1e-9)
    ]
    assert len(hits) == 1, "the tie point carries {:d} labels: {:s}".format(len(hits), repr(hits))


def test_distinct_bodies_still_keep_distinct_labels():
    """The other half of the contract: a tied interface or a crack plane has two topologically
    distinct nodes at one point, and merging those would weld the model shut."""
    mesh = AdaptiveMesh(splitFactor=2, topology=Hex20Topology())
    mesh.refine(mesh.add_root(PARENT_A, 0))
    mesh.refine(mesh.add_root(PARENT_A, 1))
    labelsOfBody = []
    for componentId in (0, 1):
        labelsOfBody.append(
            {label for label in mesh.registry.coordinates if mesh.registry.componentOf[label] == componentId}
        )
    assert labelsOfBody[0] and labelsOfBody[1]
    assert not (labelsOfBody[0] & labelsOfBody[1])
