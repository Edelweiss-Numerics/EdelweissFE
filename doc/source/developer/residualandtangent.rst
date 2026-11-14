.. _residual-and-tangent-convention:

The residual and the tangent
============================

Every entity that enters the discrete system -- an element, a load, a constraint -- contributes two things: a
vector :math:`\boldsymbol{P}` to the residual, and its tangent
:math:`\boldsymbol{K} = \partial \boldsymbol{P} / \partial \boldsymbol{U}`. The rule is the same for all of them:

* An **element** adds its internal forces,
  :math:`\boldsymbol{P} = \boldsymbol{f}_\mathrm{int} = \int_\Omega \boldsymbol{B}^T \boldsymbol{\sigma} \, \mathrm{d}V`,
  and its stiffness matrix.
* A **load** adds the *negative* of the external load it applies: a node force :math:`\boldsymbol{F}` adds
  :math:`-\boldsymbol{F}`, a body force :math:`-\int_\Omega \boldsymbol{N}^T \boldsymbol{b} \, \mathrm{d}V`. A dead
  load has no tangent; a load that follows the deformation, such as a pressure on the current surface, has one.
* A **constraint** adds the negative of the force it exerts on its nodes -- just what an element of the same
  mechanics would add as its internal force. A penalty spring of stiffness :math:`k` that pulls a degree of freedom
  :math:`u` towards :math:`\bar{u}` adds :math:`P = k\,(u - \bar{u})` and :math:`K = k`; a Lagrange multiplier
  :math:`\boldsymbol{\lambda}` enforcing :math:`\boldsymbol{g}(\boldsymbol{u}) = \boldsymbol{0}` adds
  :math:`\boldsymbol{\lambda}^T \partial \boldsymbol{g} / \partial \boldsymbol{u}` to the rows of the nodes and
  :math:`\boldsymbol{g}` to the rows of the multipliers. See :doc:`/model/constraints` for writing one.

A solver sums the contributions in three loops of every iteration: ``computeElements`` over the elements,
``assembleLoads`` over the loads -- every load assembles its own contribution, wherever it acts (see
:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`) -- and ``assembleConstraints`` over the constraints of
the discrete system (see :doc:`/model/constraints`). Summed over all entities,

.. math::

    \boldsymbol{G}(\boldsymbol{U}) = \sum \boldsymbol{P} = \boldsymbol{f}_\mathrm{int} - \boldsymbol{f}_\mathrm{ext}

is the out-of-balance force, which equilibrium makes zero, and the summed :math:`\boldsymbol{K}` is its derivative.
Newton's method replaces :math:`\boldsymbol{G}(\boldsymbol{U} + \Delta \boldsymbol{U}) = \boldsymbol{0}` by its
linearization :math:`\boldsymbol{G} + \boldsymbol{K}\, \Delta \boldsymbol{U} = \boldsymbol{0}`, so the implicit
solvers solve

.. math::

    \boldsymbol{K}\, \Delta \boldsymbol{U} = \boldsymbol{R}, \qquad
    \boldsymbol{R} = -\boldsymbol{G} = \boldsymbol{f}_\mathrm{ext} - \boldsymbol{f}_\mathrm{int}

in every iteration. The right-hand side :math:`\boldsymbol{R}` is what the convergence criteria check, and what the
multi-point-constraint condensation and the Dirichlet boundary conditions act on. The other solvers use the same
sums:

* ``NID`` subtracts the inertia and damping forces from :math:`\boldsymbol{R}`, and adds their tangents to
  :math:`\boldsymbol{K}`.
* ``NED`` accelerates the nodes with the net force :math:`\boldsymbol{R}`, and needs no tangent.
* ``NISTPArcLength`` sums the loads twice, at the start of the step (the dead load, :math:`\boldsymbol{P}_0`) and
  at its end; their difference is the reference load, :math:`\boldsymbol{P}_\mathrm{ref}`. Scaled by the load
  parameter :math:`\lambda`, it enters
  :math:`\boldsymbol{R} = -(\boldsymbol{P}_0 + \lambda \boldsymbol{P}_\mathrm{ref} + \boldsymbol{P}_\mathrm{elements}
  + \boldsymbol{P}_\mathrm{constraints})`.

Why one rule? Because then :math:`\boldsymbol{K}` is always the derivative of exactly what was added to
:math:`\boldsymbol{P}`, for every entity alike. That can be checked by finite differences, entity by entity, and the
same check catches a contribution whose sign alone is wrong: :math:`\boldsymbol{K}` then comes out as
:math:`-\partial \boldsymbol{P} / \partial \boldsymbol{U}`. A contribution negated together with its tangent passes it,
so ``tests/test_residual_tangent_convention.py`` checks every constraint this way and also checks the direction of the
forces it exerts (see :doc:`/model/constraints`); a dead load has no tangent, so there the sign of each element load is
checked against the load integrated independently.

The elements' sum alone, :math:`\boldsymbol{f}_\mathrm{int}`, is what the quasi-static solvers report as the nodal
result ``P`` of a field output, and the dynamic solvers add the inertial and damping forces to it
(:doc:`/output/nodefieldresults`): at a degree of freedom with a Dirichlet boundary condition and no load, it is the reaction
force. ``NED``'s net force :math:`\boldsymbol{R}` is its own state, and is not reported.
