Input file syntax
=================

Relevant module: ``edelweissfe.utils.inputfileparser``

EdelweissFE uses an input file syntax similar to Abaqus, using human readable input files.
The general syntax makes use of ``*keywords``, possibly with ``option=values`` in the same line and
supplemented by data lines.
In general, options and data entries are comma separated.
Indentations and blank lines have no effect.

.. code-block:: edelweiss

    *keyword, option1=value, option2=another_value
    data, data, data, data
    data, data, data, data
    data, data, data, data

    **this is a comment
    **again a comment

    *another_keyword, option1=value, option2=another_value
    data, data, data, data
    >>subkeyword, option1=value, option2=another_value
    >>subkeyword, option1=other_value

Some keywords contain ``>>subkeyword`` lines, e.g. the step actions of a ``*step``, the field outputs of a
``*fieldOutput`` or the result and configuration blocks of an ``*output``. A ``>>`` line belongs to the keyword above it and carries its
own options; most subkeywords may be repeated.


See :doc:`/developer/inputsystem` for how a keyword's grammar is declared and how to add a new one.

Each keyword declares its options in its own module in ``edelweissfe/keywords/``. All keywords and their options can be
printed using

.. code-block:: console

    edelweissfe --keywords

.. automodule:: edelweissfe.utils.inputfileparser
   :members:

Keywords
********

All keywords and their options are listed in the :doc:`keyword reference <keywords>`.

