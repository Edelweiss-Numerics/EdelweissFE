Solvers
=======

.. automodule:: edelweissfe.config.solvers
   :members:


``NIST`` - Nonlinear Implicit Static
------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstatic
   :members:

.. pprint:: solver:NIST


``NISTParallel`` - Nonlinear Implicit Static (parallel)
-------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstaticparallel
   :members:

.. pprint:: solver:NISTParallel

``NISTPArcLength`` - Nonlinear Implicit Static - Arc length
-----------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstaticparallelarclength
   :members:

.. pprint:: solver:NISTPArcLength

``NID`` - Nonlinear Implicit Dynamic (Newmark-beta)
----------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitdynamic
   :members:

.. pprint:: solver:NID

The relevant blocks of a deck, abridged from the regression case ``testfiles/marmot/NID`` -- a bar
under a suddenly applied end load, oscillating about its static deflection:

.. code-block:: edelweiss

    *solver, solver=NID, name=theSolver
    newmarkBeta=0.25
    newmarkGamma=0.5

    *fieldOutput
    >>perNode, nSet=gen_right, field=displacement, result=U, name=tipU, f(x)='np.mean(x[:,0])', saveHistory=True
    >>perNode, nSet=gen_right, field=displacement, result=V, name=tipV, f(x)='np.mean(x[:,0])', saveHistory=True
    >>perNode, nSet=gen_right, field=displacement, result=A, name=tipA, f(x)='np.mean(x[:,0])', saveHistory=True

    *step, solver=theSolver
    stepLength=2.0, startInc=0.01, maxInc=0.01, minInc=1e-4, maxNumInc=1000, maxIter=25
    >>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0
    >>nodeforces, name=endLoad, nSet=gen_right, field=displacement, 1=0.0987, f(t)='1'

The time increment is the step's increment (``stepLength`` times the step-progress increment), so a
constant ``dt`` is obtained with ``startInc = maxInc``; a cutback shrinks it like any other implicit
increment. The velocity and acceleration are available to every ``*fieldOutput`` as ``result=V`` and
``result=A`` on the displacement field, and travel with the ``*output, type=restart`` checkpoints,
which is what makes a resumed run continue the same trajectory rather than restart it from rest.

The material must report a density (for ``LinearElastic``, the third material parameter): the mass
is assembled from it, and a time-integrated degree of freedom that receives none is refused rather
than integrated as though it had inertia.

A ``*modelModifier, type=hAdaptivity`` refining the mesh mid-step is supported: ``V`` and ``A`` are
carried onto new nodes by the node-field warm start, the mass and damping are reassembled on the
refined mesh, and the acceleration is re-solved from equilibrium on the increment that follows --
see :doc:`modelmodifiers`, "Refinement under a dynamic solver", for what is conserved exactly across
such an event and what is not.

Verified in ``tests/test_nid_newmark.py``, ``tests/test_nid_restart.py`` and
``tests/test_nid_amr.py`` against the exact discrete Newmark recurrence of the same problem, against
the continuous closed form at second order, for exact conservation of the discrete energy, for
restart equivalence of ``U``, ``V`` and ``A``, and -- under live refinement -- for exact transfer of
a velocity field of degree at most one onto new nodes together with exact conservation of mass,
momentum and kinetic energy in that case.

``NEST`` - Nonlinear Explicit Static
-------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitstatic
   :members:

.. pprint:: solver:NEST

``NESTParallel`` - Nonlinear Explicit Static (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitstaticparallel
   :members:

.. pprint:: solver:NESTParallel

``NED`` - Nonlinear Explicit Dynamic
-------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamic
   :members:

.. pprint:: solver:NED

Rebuilding the equation system during a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit analysis with contact rebuilds its equation system while it runs: a contact search
re-assigns which nodes a constraint couples, which changes that constraint's degree-of-freedom
footprint, and everything sized by the footprints has to follow.

Such a change is much narrower than it looks. It touches no node, no field, no scalar variable and
no element, so the degree-of-freedom numbering and every element's indices stay exactly as they
were -- and with them the lumped inertia, its inverse, the mass-proportional damping rate and the
multi-point-constraint transformation, none of which is a function of the constraints' footprints.
The solver therefore re-locates the constraints alone
(:meth:`~edelweissfe.numerics.dofmanager.DofManager.refreshConstraintIndices`) and keeps those
operators, rather than constructing a new :class:`~edelweissfe.numerics.dofmanager.DofManager` and
assembling them again over every element.

That shortcut is only valid while everything the operators *are* a function of is unchanged, so it
is taken only after checking the element set, the multi-point constraints, the node count and the
scalar variables against what was recorded when they were last assembled, and only when no step
action changes a material property mid-step. Anything that does not match falls back to a full
build. A topology change -- a mesh refinement -- is not a connectivity change and never takes this
path at all: it builds the system from scratch.

``NEDParallel`` - Nonlinear Explicit Dynamic (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamicparallel
   :members:

.. pprint:: solver:NEDParallel
