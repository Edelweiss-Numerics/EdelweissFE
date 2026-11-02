Node field results
==================

Every node field (:class:`~edelweissfe.fields.nodefield.NodeField`) holds the same five entries, from its
construction on: ``U``, ``dU``, ``P``, ``V`` and ``A``
(:attr:`~edelweissfe.fields.nodefield.NodeField.ENTRIES`). There are no others, and no code creates one. These are
the names a field output asks for with ``result=``:

.. code-block:: edelweiss

    *fieldOutput
    >>perNode, nSet=bottom, field=displacement, result=P, name=RF

Each entry has one meaning, the same in every solver. This page defines it.

Who owns what
-------------

One rule decides where a value lives: *the model owns the state the next attempt starts from; a solver owns only
what lives within one equation system.* An entry is therefore of one of two kinds:

- **state** -- what a solver reads back from the model whenever it (re)builds its equation system, and what a
  restart checkpoint must carry for the run to continue;
- **result** -- what a solver writes for output and for the user; no solver reads it back.

Iterations work on the solver's vectors only. A cut back increment writes nothing. An accepted increment writes
every entry, once. A rebuilt equation system (a changed mesh, a changed active domain, a changed coupling) reads
the state entries from the model. A refinement of the mesh interpolates the state entries onto the nodes it
creates; the results there are zero until the next accepted increment.

Every solver passes these entries through the model's layout of its equation system
(:class:`~edelweissfe.models.equationsystemlayout.EquationSystemLayout`), and in no other way:
:meth:`~edelweissfe.models.equationsystemlayout.EquationSystemLayout.readStateFromModel` when it (re)builds the
system, :meth:`~edelweissfe.models.equationsystemlayout.EquationSystemLayout.writeResultsToModel` when it accepts
an increment.

The entries
-----------

.. list-table::
   :header-rows: 1
   :widths: 8 8 54 30

   * - Entry
     - Kind
     - Meaning
     - Written
   * - ``U``
     - state
     - The total solution: the displacement, the rotation, the nonlocal damage, ... of the field.
     - at acceptance
   * - ``V``, ``A``
     - state
     - The rate and the second rate of the solution at the accepted time: velocity and acceleration of a field with
       inertia. Zero for a field without inertia, and after a quasi-static step, which leaves the model at rest. The
       explicit solver integrates the velocity staggered by half an increment: its ``V`` is
       :math:`\boldsymbol{v}_{n+1/2}`.
     - at acceptance; the implicit dynamic solver completes ``A`` at the start of an increment
   * - ``dU``
     - result
     - The increment of the solution in the last accepted increment.
     - at acceptance
   * - ``P``
     - result
     - The internal force, with the inertial and damping forces of a dynamic analysis: the residual contributions of
       the elements (and the other computed entities) and of the time integration, *without* loads and constraints
       (see :ref:`residual-and-tangent-convention`). At a support that no load acts on, it is the support reaction;
       summed over the nodes of a support, it is the reaction force. Zero at a node without element.
     - at acceptance

The Lagrange multipliers of constraints are scalar variables, not node field entries
(:class:`~edelweissfe.variables.scalarvariable.ScalarVariable`). They are state as well: each holds its total
value, written at acceptance and read back at every rebuild. A constraint that is switched off keeps the value it
had.

Models whose equation system covers a part of their nodes
---------------------------------------------------------

A meshfree model (EdelweissMeshfree) has nodes that carry no total solution: the nodes of the background grid of
the material point method and the nodes of the kernel functions of RKPM are *incremental* -- their degrees of
freedom are the coefficients of the increment, starting at zero in every increment, while the particles and
material points carry the state. On these nodes, ``U`` is zero and ``dU`` holds the coefficients of the last
accepted increment, which is their meaningful nodal value. Nodes outside the active domain of an increment hold
zero in every entry.

Using the entries in code
-------------------------

The entries exist from the construction of a node field on, also in a model built programmatically; a resize (for
instance, by adaptive mesh refinement) keeps all of them. Reading or writing an entry through the
:class:`~edelweissfe.numerics.dofmanager.DofManager` refuses a name that is not one of the five, and a node field
the DofManager does not number: a node field is read or written as a whole only by the DofManager that numbers it,
since its rows are addressed by position; on a node set, its rows are found by node, so that any node field of a
numbered field can be read or written there (e.g. the node field of the model, while the equation system numbers
that of an active domain). A restart checkpoint holds every entry of every node field, and a
checkpoint lacking one is refused (see :doc:`restart`).

Where the solvers do not follow this page yet
---------------------------------------------

The meanings above are being made to hold in every solver. Until then, these deviations remain; each is removed
from this list together with the deviation:

- The explicit solver (``NED``) writes as ``P`` the net force that accelerates the nodes,
  :math:`\boldsymbol{f}_\mathrm{ext} - \boldsymbol{f}_\mathrm{int}` including loads and constraints -- the opposite
  sign of the internal force -- and reads it back when it builds its equation system.
- The implicit dynamic solver (``NonlinearImplicitDynamic``) writes as ``P`` the internal force without the inertial
  and damping forces, and its zero increment at a step start writes ``P`` as zero.
- A quasi-static step after a dynamic one leaves ``V`` and ``A`` of the last dynamic increment in the model.
- In a meshfree model, ``U`` at the incremental nodes is the sum of their increments since the equation system was
  last built, and the explicit meshfree solver writes no node field entry.
