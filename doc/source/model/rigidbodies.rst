Rigid bodies
============

A rigid body couples the motion of many surface (visualization) nodes to a single *reference point*
(RP). The surface nodes are **not** independent degrees of freedom: they carry no field variables
and are never written to a node field. Instead, their current configuration is a pure function of
the RP's kinematics, so the only unknowns a rigid body contributes to the global system are the RP
displacement and (in 3D) rotation DOFs. Rigid bodies are consumed by contact constraints such as
:doc:`surfacetodiscreterigidbodypenalty </model/constraints>` and are rendered by the output managers via
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
respect to :math:`\boldsymbol{\theta}_{RP}` requires the right Jacobian of the exponential map,

.. math::
    \mathbf{J}_r(\boldsymbol{\theta}) = \mathbf{I}
    - \frac{1-\cos\varphi}{\varphi^2}\,\mathrm{skew}(\boldsymbol{\theta})
    + \frac{\varphi-\sin\varphi}{\varphi^3}\,\mathrm{skew}(\boldsymbol{\theta})^2 ,
    \qquad \varphi = \lVert \boldsymbol{\theta} \rVert :

:math:`\mathbf{R}\,\mathbf{J}_r` maps a perturbation of the stored pseudo-vector onto the spatial rotation it
produces, the identity only at zero rotation (this is what makes the tangent of the contact against a discrete
rigid body geometrically exact -- see :ref:`integrated-contact-rigid-body`).

A surface node with initial position offset :math:`\boldsymbol{\rho}_i =
\mathbf{X}_i - \mathbf{X}_{RP}` relative to the RP moves rigidly to the current configuration

.. math::
    \mathbf{x}_i = (\mathbf{X}_{RP} + \mathbf{u}_{RP}) + \mathbf{R}\,\boldsymbol{\rho}_i .

:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.updateKinematics` writes these
current coordinates onto the surface nodes after every converged increment, which is what the output
managers read to draw the moving body. Because the surface nodes are not DOFs, this is a display
convenience, not part of the solve: every nonlinear solver invokes it through
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateRigidBodies`, so all
solver variants (serial, parallel, arc-length, explicit) keep the visualization geometry consistent.
Contact constraints do not rely on it -- they take the RP pose directly from the current Newton iterate
(see below).
That pose is provided by
:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.poseFromDofs`. The integrated
contact constraint searches the body-frame triangles of
:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.referenceTriangles`, which
requires a closed surface without zero-area triangles, so that every triangle has a defined outward
normal.

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

The rigid surface
-----------------

Contact with a discrete rigid body (``surfaceToDiscreteRigidBodyPenalty``) pairs its contact points with the triangles
of the rigid surface, in the body frame, with their outward normals
(:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.referenceTriangles`); the body's pose maps them
into the current configuration. They are the cells of the outer surface of the mesh, extracted once
(:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.outerSurface`). Outward normals are established
from the mesh topology (``compute_normals(..., auto_orient_normals=True)``) rather than trusting the source file's face
winding, so they are reliable for any correctly closed input mesh; a surface that is not closed, not made of
triangles, or holds a triangle of zero area is refused.

Creating a discrete rigid body
------------------------------

Discrete rigid bodies are normally created from a surface mesh file (Exodus / STL / OBJ) with the
``discreterigidbodygenerator`` model generator, which loads the mesh, creates the RP and surface
nodes, computes the mass properties, and registers the body in ``model.rigidBodies``. See
:doc:`/model/generators` for the generator options and an input-file example.

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
