Journal
=======

Relevant module: ``edelweissfe.journal.journal``

The journal writes two kinds of output, in two columns: the message on the left, its sender on the right.

Messages
    :meth:`~edelweissfe.journal.journal.Journal.message` is for text. Lines longer than the message column are
    wrapped, and only the first line carries the sender.

Preformatted lines
    :meth:`~edelweissfe.journal.journal.Journal.preformatted` is for lines whose layout carries meaning, such as
    the rows of a table. They are never wrapped: a line longer than the message column pushes the sender to the
    right instead. Every line carries the sender. The iteration table of the implicit solvers is written this
    way, see :mod:`edelweissfe.solvers.base.convergencetable`.

The level of a message is its nesting depth in the log, and it follows where the message is written:

Level 0
    The job: model setup and summaries.
Level 1
    An increment: its header, the convergence verdict, cutbacks, monitors, model modifiers such as adaptive
    refinement, and restart checkpoints.
Level 2
    Inside an increment: equation-system rebuilds, the iteration table, linear solver output, contact status.

A warning keeps the level of the place it is raised and is marked ``WARNING:`` or ``⚠`` instead of being
moved out to level 0. The implicit solvers open each increment's block with a separator line before the topology
update, so refinement and contact output head the increment they belong to.

.. automodule:: edelweissfe.journal.journal
   :members:
