Rigid bodies
============

A rigid body couples the motion of many surface (visualization) nodes to a single *reference point*
(RP). The surface nodes are **not** independent degrees of freedom: they carry no field variables
and are never written to a node field. Instead, their current configuration is a pure function of
the RP's kinematics, so the only unknowns a rigid body contributes to the global system are the RP
displacement and (in 3D) rotation DOFs. Rigid bodies are consumed by contact constraints such as
:doc:`nodetodiscreterigidbodypenalty <constraints>` and are rendered by the output managers via
their surface nodes.

.. contents::
    :local:

Kinematics
----------

The reference point owns a total displacement :math:`\mathbf{u}_{RP}` and, in 3D, a rotation
pseudo-vector :math:`\boldsymbol{\theta}_{RP}` (an axis-angle vector: direction = rotation axis,
magnitude = rotation angle in radians). The orientation is recovered through the SO(3) exponential
map (Rodrigues' formula)

.. math::
    \mathbf{R}(\boldsymbol{\theta}) = \exp(\mathrm{skew}(\boldsymbol{\theta}))
    = \mathbf{I}
    + \frac{\sin\varphi}{\varphi}\,\mathrm{skew}(\boldsymbol{\theta})
    + \frac{1-\cos\varphi}{\varphi^2}\,\mathrm{skew}(\boldsymbol{\theta})^2 ,
    \qquad \varphi = \lVert\boldsymbol{\theta}\rVert ,

where :math:`\mathrm{skew}(\mathbf{v})` is the skew-symmetric matrix with
:math:`\mathrm{skew}(\mathbf{v})\,\mathbf{x} = \mathbf{v}\times\mathbf{x}`. The stored DOF is the
*raw total* pseudo-vector; because :math:`\exp(\boldsymbol{\theta}+\delta\boldsymbol{\theta}) \neq
\exp(\boldsymbol{\theta})\exp(\delta\boldsymbol{\theta})` in general, linearizing any quantity with
respect to :math:`\boldsymbol{\theta}_{RP}` requires the right Jacobian
:math:`\mathbf{J}_r(\boldsymbol{\theta}_{RP})` of the exponential map (this is what makes the
discrete-rigid-body contact tangent geometrically exact — see :doc:`constraints`).

A surface node with initial position offset :math:`\boldsymbol{\rho}_i =
\mathbf{X}_i - \mathbf{X}_{RP}` relative to the RP moves rigidly to the current configuration

.. math::
    \mathbf{x}_i = (\mathbf{X}_{RP} + \mathbf{u}_{RP}) + \mathbf{R}\,\boldsymbol{\rho}_i .

:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.updateKinematics` writes these
current coordinates onto the surface nodes after every converged increment, which is what the output
managers read to draw the moving body. Because the surface nodes are not DOFs, this is a display /
broadphase convenience, not part of the solve: every nonlinear solver invokes it through
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateRigidBodies`, so all
solver variants (serial, parallel, arc-length, explicit) keep the visualization geometry consistent.
Contact constraints do not rely on it — they query the surface with the RP pose taken directly from
the current Newton iterate (see below).

Mass properties
---------------

For dynamics (and for reporting), the exact volume, mass, center of mass and inertia tensor of a
homogeneous solid bounded by the (closed, manifold) surface mesh are computed by the
:mod:`~edelweissfe.utils.polyhedronmassproperties` module. Each surface facet is fan-triangulated
and, together with the origin, forms a signed tetrahedron; the divergence theorem then turns the
volume integrals of :math:`1`, :math:`\mathbf{x}`, and :math:`\mathbf{x}\otimes\mathbf{x}` into a
sum over those tetrahedra, e.g. for the volume

.. math::
    V = \int_\Omega \mathrm{d}V
      = \frac{1}{6}\sum_{\text{facets}} \mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}) ,

with :math:`\mathbf{a},\mathbf{b},\mathbf{c}` the facet's triangle vertices. Consistent outward face
winding makes the signed tetrahedra cancel correctly for non-convex bodies, so the result is exact
(no volumetric mesh required). The inertia tensor is returned about the center of mass, in the
global (non-rotated) axes of the input vertices.

Surface query
-------------

Contact needs, for an arbitrary array of world-frame query points, the signed distance to the rigid
surface (negative inside) and the outward normal of the closest facet, evaluated for the body's
current pose. This is provided by :class:`~edelweissfe.utils.discretesurfacequery.DiscreteSurfaceQuery`,
which wraps VTK's ``vtkImplicitPolyDataDistance`` and a static cell locator built once on the
*reference* mesh. Rather than re-transform the whole mesh each query, the query points are mapped
into the body's local (reference) frame,

.. math::
    \mathbf{p}_{\text{local}} = \mathbf{X}_{RP}
        + \mathbf{R}^{\top}\big(\mathbf{p} - (\mathbf{X}_{RP}+\mathbf{u}_{RP})\big) ,

the distance/closest-facet lookup is done there, and the returned normals are rotated back to the
world frame by :math:`\mathbf{R}`.

.. note::
    The implementation stores points as *row* vectors (shape ``(N, 3)``), for which
    ``v.dot(M)`` computes :math:`\mathbf{M}^{\top}\mathbf{v}` per row. Consequently the inverse
    map above is coded as ``localCoords.dot(rotation_matrix)`` (i.e. :math:`\mathbf{R}^{\top}`) and
    the forward map of normals as ``normals.dot(rotation_matrix.T)`` (i.e. :math:`\mathbf{R}`) —
    the opposite of what a naive read of the transpose suggests. This convention is documented in
    the source and was verified numerically against a known rotated mesh; do not "fix" the transpose
    without re-deriving it.

Outward normals are established from the mesh topology (``compute_normals(..., auto_orient_normals=True)``)
rather than trusting the source file's face winding, so the sign of the signed distance and the
direction of the returned normals are reliable for any correctly closed input mesh.

Creating a discrete rigid body
------------------------------

Discrete rigid bodies are normally created from a surface mesh file (Exodus / STL / OBJ) with the
``discreterigidbodygenerator`` model generator, which loads the mesh, creates the RP and surface
nodes, computes the mass properties, and registers the body in ``model.rigidBodies``. See
:doc:`generators` for the generator options and an input-file example.

API
---

Rigid body base class
^^^^^^^^^^^^^^^^^^^^^^^

Module ``edelweissfe.rigidbodies.rigidbody``

.. autoclass:: edelweissfe.rigidbodies.rigidbody.RigidBody
    :members:

Discrete rigid body
^^^^^^^^^^^^^^^^^^^^

Module ``edelweissfe.rigidbodies.discreterigidbody``

.. autoclass:: edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody
    :members:

Polyhedron mass properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module ``edelweissfe.utils.polyhedronmassproperties``

.. automodule:: edelweissfe.utils.polyhedronmassproperties
    :members:

Discrete surface query
^^^^^^^^^^^^^^^^^^^^^^^

Module ``edelweissfe.utils.discretesurfacequery``

.. autoclass:: edelweissfe.utils.discretesurfacequery.DiscreteSurfaceQuery
    :members:
