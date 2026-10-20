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

.. automodule:: edelweissfe.journal.journal
   :members:
