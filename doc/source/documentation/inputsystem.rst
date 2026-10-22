The input system: how ``.inp`` grammar is defined
===================================================

This page explains, for developers, how EdelweissFE turns a keyword in an ``.inp`` file (say
``*dirichlet`` or ``*output, type=ensight``) into a running Python object -- and, more importantly,
how you add a *new* keyword or a new pluggable module (a material, an output manager, a step
action, ...) correctly.

There is exactly **one** mechanism for this. Every keyword's grammar -- its options, their types,
defaults, required-ness, nested ``>>`` blocks -- is declared once, as an ordinary Python
:mod:`dataclasses` class living next to the code it configures. That single declaration is then
reused for everything: parsing, validation, error messages, the rendered documentation on this
page, and the ``--keywords`` command-line summary. There is no second, hand-maintained grammar
description anywhere to keep in sync.

The four layers
----------------

Reading top to bottom, from "how a developer calls this from Python" down to "how a line of text
reaches that call":

.. list-table::
   :widths: 8 20 40
   :header-rows: 1

   * - Layer
     - Name
     - What it is
   * - **L1**
     - Typed Python API
     - An ordinary, typed ``__init__`` (or factory function). This is the *real* class -- a
       :class:`~edelweissfe.stepactions.dirichlet.StepAction`, an
       :class:`~edelweissfe.outputmanagers.ensight.OutputManager` -- constructible directly from
       Python, with no knowledge that an ``.inp`` file exists.
   * - **L2**
     - Option schema
     - A frozen ``dataclass`` describing what that L1 constructor's ``.inp``-facing options are:
       their names, Python types, defaults and required-ness. Pure data, no parsing logic.
   * - **L3**
     - Registry
     - A ``(category, name) -> class`` lookup table (``"outputmanager", "ensight"`` ->
       ``OutputManager``), populated from a built-in table and from third-party
       ``importlib.metadata`` entry points.
   * - **L4**
     - Adapter
     - The (usually short) piece of code that turns a parsed ``.inp`` block into a call to the L1
       constructor, using the L2 schema to validate/coerce and the L3 registry to find the right
       class.

The key idea is that **L2 is the single source of truth**. L3 exists so a name in an ``.inp`` file
can find its class without an import graph reaching into every module up front. L4 exists because
an ``.inp`` file is text and an L1 constructor wants real Python values (a ``NodeSet`` object, not
the string ``"bottom"``) -- something has to bridge that gap, and the schema is what it validates
against.

If you remember one rule: **the schema is the grammar.** Nothing else defines what a keyword
accepts. There is no separate "keyword definition" object, no dict of allowed options maintained
next to the schema -- change the schema, and parsing, validation, error messages and documentation
all follow immediately.

L2: declaring a schema
-----------------------

Schemas are built with three field-declaring helpers from :mod:`edelweissfe.utils.schema`, used as
ordinary dataclass field values:

``schemaField(description=..., dtype=..., default=...)``
    An ordinary ``key=value`` option on the keyword's own line.

``subKeywordField(description=..., schema=...)``
    A repeatable ``>>subkeyword`` block, with its own nested schema. The field holds a *tuple* of
    block instances (possibly empty), since a block kind may be repeated or omitted.

``datalineField(description=...)``
    Marks "this keyword also has a body of raw datalines below its option line" (element
    connectivity, node coordinates, material property rows, ...). It records only presence and
    documentation -- interpreting the actual lines is the owning class's job (see
    :class:`~edelweissfe.utils.schema.DatalineAggregatingSchema` below).

A minimal example, ``*nSet``:

.. code-block:: python

    @dataclass(frozen=True)
    class NSetSchema:
        nSet: str | None = schemaField(description="name", dtype=str, default=None, required=True)
        generate: bool = schemaField(
            description="set True to generate from data line 1: start-node, end-node, step",
            dtype=bool,
            default=False,
        )
        datalines: list | None = datalineField(description="Abaqus like node set definition lines", required=True)

A field is required exactly when it has neither ``default=`` nor ``default_factory=`` -- you don't
set ``required=True`` by hand for the common case.

A more involved example -- ``dirichlet``'s schema declares a numbered-component option
(``1=``...``6=``, not valid Python identifiers) via ``optionName``, and marks ``nSet`` /
``analyticalField`` as :attr:`~edelweissfe.utils.schema.SchemaFieldMeta.structuralOnly` because
they name *existing model objects* rather than plain values (more on this below):

.. code-block:: python

    @dataclass(frozen=True)
    class DirichletSchema:
        nSet: str | None = schemaField(
            description="The node set for application of the boundary condition.",
            dtype=str, default=None, required=True, structuralOnly=True,
        )
        field: str | None = schemaField(
            description="Field for which the boundary condition is active.", dtype=str, default=None, required=True
        )
        component1: float | None = schemaField(
            description="Prescribe first component of field.", dtype=float, default=None, optionName="1"
        )
        # ... component2..6 ...
        f_t: str | None = schemaField(
            description="Define an amplitude in the step progress interval [0...1]",
            dtype=str, default=None, optionName="f(t)",
        )

Field flags you'll actually need
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most schemas need nothing beyond ``description``/``dtype``/``default``/``required``. Four flags
cover the remaining, real cases -- each one only changes how the field is *rendered* or *validated
against*, never the schema's own field type:

``structuralOnly=True``
    This option names an existing model object (a node/element set, a material, a surface) that an
    L4 adapter resolves and pops *before* the rest of the definition reaches the schema. It is
    never actually seen by :func:`~edelweissfe.utils.schema.buildSchemaFromOptions` -- it exists on
    the schema purely so the option is documented in the rendered grammar. See ``dirichlet``'s
    ``nSet`` above: the adapter turns the string ``"bottom"`` into ``model.nodeSets["bottom"]``
    itself, and by the time ``buildSchemaFromOptions`` runs, ``nSet`` is already gone from the
    mapping.

``optionsOverrideOnly=True``
    The mirror image: an option that is *never* set on the keyword's own line or in a ``>>`` block,
    only later via a dynamic ``>>options, name=X, ...`` override (see below). Ensight's
    ``intermediateSaveInterval`` is the example -- excluded from the keyword's own rendered block,
    but still validated when an ``>>options`` block later touches it.

``updateOnly=True``
    An option that belongs only to a keyword's ``update<keyword>`` partial re-declaration (e.g.
    ``updateDirichlet``), not the base keyword's initial declaration.

``documentedDefault=...``
    Overrides only the default value *shown in the rendered documentation*, for the rare case where
    it must differ from the field's real runtime default (used to record a legacy default without
    changing behavior). You will almost never need this.

Interpreting datalines
~~~~~~~~~~~~~~~~~~~~~~~

A :func:`~edelweissfe.utils.schema.datalineField` only says *that* a keyword has a dataline body;
something still has to turn those raw lines into data. The default L4 adapter treats each dataline
as one independent instance sharing the flat option schema -- correct for most keywords. When a
keyword's datalines are genuinely heterogeneous (each line picking its own "kind", like
``meshplot``'s ``create=perNode|perElement|xyData``), the schema instead derives from
:class:`~edelweissfe.utils.schema.DatalineAggregatingSchema` and implements
``fromDatalines(cls, datalines)``, building *one* instance from *all* datalines of the block at
once. Prefer the default (a plain flat schema); reach for ``DatalineAggregatingSchema`` only for
that aggregate shape.

L3: the registry
------------------

:mod:`edelweissfe.config.registry` maps ``(category, name)`` to a class, lazily. Categories are
things like ``"outputmanager"``, ``"stepaction"``, ``"constraint"``, ``"material"``, ``"keyword"``.
Two things matter in practice:

- **Names and categories are case-insensitive.** ``registry.lookup("outputmanager", "Ensight")``
  and ``registry.lookup("OUTPUTMANAGER", "ensight")`` are identical.
- **Nothing is imported until it's actually looked up.** The built-in table is a plain dict of
  strings (``"edelweissfe.outputmanagers.ensight:OutputManager"``); the target module is only
  ``import``-ed the first time that name is requested.

.. code-block:: python

    from edelweissfe.config import registry

    outputManagerClass, outputManagerSchema = registry.lookup("outputmanager", "ensight")

The schema returned alongside the class is simply that class's own ``schema`` attribute (see
:class:`~edelweissfe.utils.schema.OptionSchemaProvider` below) -- the registry does not maintain a
second copy of it.

**Registering a new built-in** is one line in ``config/registry.py``'s ``_BUILTINS`` table (or, for
the uniform "one module per name, fixed attribute name" categories, one entry in the relevant
``_addBuiltins(...)`` call). **A third-party package** (a plugin, EdelweissMeshfree) instead
contributes via a ``pyproject.toml`` entry point, with no change to EdelweissFE at all:

.. code-block:: toml

    [project.entry-points."edelweissfe.plugins"]
    "outputmanager.myoutputmanager" = "mypackage.mymodule:MyOutputManager"

Either way, a class becomes registrable simply by deriving from
:class:`~edelweissfe.utils.schema.OptionSchemaProvider` and setting ``schema = MySchema`` as a
class attribute -- that is the *entire* contract between a module and the registry.

L4: from parsed text to a Python call
---------------------------------------

There are two shapes of L4 adapter in the codebase, depending on whether construction needs
anything beyond "validate options, call the constructor":

**The generic adapter** -- used for output managers, sections, analytical fields, generators, and
similar -- lives centrally in ``edelweissfe/helpers/inputfilehelpers.py``. It does exactly three
things per block: look the type up in the registry, validate/coerce the parsed options against the
schema it got back, and call the constructor:

.. code-block:: python

    outputManagerClass, outputManagerSchema = registry.lookup("outputmanager", outputManagerType)

    configuration = buildSchemaFromOptions(
        outputManagerSchema,
        kwargs,                # this block's key=value options, as parsed
        moduleOptions,          # this block's >> sub-keyword options, as parsed
    )

    outputManagers.append(
        outputManagerClass(outputManagerName, model, fieldOutputController, journal, plotter,
                            configuration=configuration)
    )

:func:`~edelweissfe.utils.schema.buildSchemaFromOptions` does the real work: it resolves
case-insensitive keys, coerces each value to its field's declared type, fills in defaults, and
raises a descriptive :class:`ValueError` for an unknown option, a bad value, or a missing required
one. If the schema declares :func:`~edelweissfe.utils.schema.subKeywordField` fields, matching
``>>`` blocks are built recursively into their own nested schema instances.

**The per-class hook** -- used by step actions and constraints, whose construction needs to
*resolve* something (turn an ``nSet`` name into an actual ``NodeSet`` object) before or instead of
building the schema -- is a classmethod each module overrides:
:meth:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase.fromStepActionDefinition` /
``fromConstraintDefinition``. The default implementation just forwards the raw dict (for a module
that has not been given a typed constructor yet); overriding it together with a typed ``__init__``
is how a module opts in. ``dirichlet`` is the canonical example:

.. code-block:: python

    class StepAction(DirichletBase):
        schema = DirichletSchema

        def __init__(self, name, nSet, field, prescribedComponents, model, journal,
                     f_t=None, analyticalField=None):
            ...  # a real, typed constructor -- nSet is a NodeSet, not a string

        @classmethod
        def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
            definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
            definition.pop("name", None)
            nSetName = definition.pop("nSet")                      # structuralOnly: resolved here
            analyticalFieldName = definition.pop("analyticalField")
            configuration = buildSchemaFromOptions(cls.schema, definition)

            return cls(
                name,
                model.nodeSets[nSetName],
                configuration.field,
                cls._prescribedComponentsFromDefinition(configuration, ...),
                model, journal,
                f_t=amplitudeFromExpression(configuration.f_t),
                analyticalField=model.analyticalFields[analyticalFieldName] if analyticalFieldName else None,
            )

Notice the split of responsibility: ``fromStepActionDefinition`` pops exactly the
``structuralOnly`` fields (``nSet``, ``analyticalField``) before handing the rest to
``buildSchemaFromOptions`` -- which is precisely why those two fields are marked
``structuralOnly`` on the schema in the first place: it documents that they're resolved here, not
validated as plain strings.

This is also the seam that lets EdelweissMeshfree (or a script) construct a
:class:`~edelweissfe.stepactions.dirichlet.StepAction` directly, in Python, with a real
:class:`~edelweissfe.sets.nodeset.NodeSet` -- with no ``.inp`` file, parser, or dict in sight. The
``.inp`` front-end is a serialization of the Python API, not a second way of building the object.

Top-level keywords
--------------------

Everything above describes a *pluggable* module -- something dispatched by a ``type=`` (or
similar) option on a keyword, of which there may be many kinds (``*output, type=ensight`` vs.
``*output, type=monitor``). The ``.inp`` file's actual top-level keywords (``*element``, ``*nSet``,
``*job``, ``*output`` itself, ...) are a small, closed set, each represented by its own
:class:`~edelweissfe.keywords.base.keywordbase.KeywordBase` subclass, registered under the
``"keyword"`` category:

.. code-block:: python

    class NSetKeyword(KeywordBase):
        schema = NSetSchema
        keywordName = "nSet"              # exact spelling as written after '*' in the .inp file
        keywordDescription = "definition of a node set"

``keywordName``/``keywordDescription`` are the single source of truth for a keyword's spelling and
summary -- the parser and the rendered documentation both read them off the class, so they cannot
drift apart. A keyword whose grammar is completed by a further dispatch (``*output, type=...``)
declares no schema of its own beyond the dispatch option itself; the parser looks the dispatch
value up in the relevant registry category (``"outputmanager"`` for ``type=``, ``"generator"`` for
``*modelGenerator``'s ``generator=``, ...) to find the rest of the grammar.

Dynamic ``>>options`` overrides
---------------------------------

One keyword, ``>>options``, is deliberately validated *dynamically* rather than against a
pre-declared schema, because it can target any already-declared solver or output manager by name:

.. code-block:: edelweiss

    *solver, name=mySolver, solver=NIST
    *step, solver=mySolver
    >>options, name=mySolver, extrapolation=linear

At parse time only ``name`` is required; every other ``key=value`` pair is accepted unvalidated.
Once the step action runs, it resolves ``name`` against ``model.solvers``/``model.outputManagers``,
then validates the options actually given against *that instance's own*
``type(target).schema`` via :func:`~edelweissfe.utils.schema.coercePresentOptions` (the
partial-application counterpart of ``buildSchemaFromOptions`` -- no missing-required check, since
an override is by definition partial), and applies them via ``target.applyOptionsOverride(...)``.
This is why a field meant to be reachable only this way is marked
``optionsOverrideOnly=True`` on its schema, as described above.

Documentation and the grammar surface come from the same schema
-------------------------------------------------------------------

Because a schema is the single source of truth, the reference documentation on the other pages
under :doc:`index` is not hand-written prose about options -- it is rendered directly from the same
schema classes described above, via the ``.. pprint::`` Sphinx directive:

.. code-block:: rst

    .. pprint:: outputmanager:ensight
        :caption: Options:

``.. pprint:: category:name`` looks the target up in the L3 registry (exactly like
``registry.lookup(category, name)``) and renders an option table straight from its L2 schema:
type, default, required-ness and description, including nested ``>>`` sub-keyword blocks. **If you
add or change a schema field, its documentation updates automatically the next time the docs are
built -- there is nothing further to write by hand.** The only thing you, as the author of a new
module, must still write is the prose description in the module's own docstring (rendered above the
table via ``.. automodule:: ... :members: __doc__``) and, ideally, one example ``.. literalinclude::``
of a working ``.inp`` snippet.

Worked example: adding a new output manager
-----------------------------------------------

Putting the pieces together, adding a new pluggable module -- say an output manager called
``myoutput`` -- looks like this:

1. **Write the L2 schema** next to the class, as a frozen dataclass:

   .. code-block:: python

       @dataclass(frozen=True)
       class MyOutputSchema:
           interval: int = schemaField(description="Export every N increments.", dtype=int, default=1)

2. **Write the L1 class**, deriving from the category's base class (here
   :class:`~edelweissfe.outputmanagers.base.outputmanagerbase.OutputManagerBase`, itself an
   :class:`~edelweissfe.utils.schema.OptionSchemaProvider`), with a typed constructor and
   ``schema = MyOutputSchema`` as a class attribute:

   .. code-block:: python

       class OutputManager(OutputManagerBase):
           schema = MyOutputSchema

           def __init__(self, name, model, fieldOutputController, journal, plotter, *,
                        configuration: MyOutputSchema = MyOutputSchema()):
               self.interval = configuration.interval
               ...

3. **Register it** -- a built-in gets one entry in ``config/registry.py``'s ``_BUILTINS`` table (or
   an ``_addBuiltins(...)`` list entry); a third-party package instead declares an
   ``edelweissfe.plugins`` entry point in its own ``pyproject.toml``. Nothing else changes: the
   generic L4 adapter in ``helpers/inputfilehelpers.py`` already knows how to drive any
   ``outputmanager``-category class through ``buildSchemaFromOptions``.

4. **Document it** by adding a short section to ``doc/source/documentation/output.rst``, following
   the same pattern every other entry on that page uses: an ``automodule`` directive targeting
   ``edelweissfe.outputmanagers.myoutput`` with ``:members: __doc__`` (to render the module's own
   docstring), followed by a ``pprint`` directive targeting ``outputmanager:myoutput`` (to render
   the options table from :class:`MyOutputSchema`) -- see the ``ensight`` section above for the
   exact two directives to copy.

That's it -- no parser change, no grammar dict to update, no golden file to regenerate by hand.
The schema you wrote in step 1 is simultaneously the validation logic, the error messages, and the
rendered documentation.

Quick reference
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - I want to...
     - Look at
   * - Declare a scalar option
     - :func:`edelweissfe.utils.schema.schemaField`
   * - Declare a repeatable ``>>`` block
     - :func:`edelweissfe.utils.schema.subKeywordField`
   * - Declare a dataline body
     - :func:`edelweissfe.utils.schema.datalineField`,
       :class:`edelweissfe.utils.schema.DatalineAggregatingSchema`
   * - Document (without validating) a structural argument
     - ``structuralOnly=True`` on :func:`~edelweissfe.utils.schema.schemaField`
   * - Make an option reachable only via ``>>options``
     - ``optionsOverrideOnly=True``
   * - Build a schema instance from parsed options
     - :func:`edelweissfe.utils.schema.buildSchemaFromOptions`
   * - Partially override an existing instance's options
     - :func:`edelweissfe.utils.schema.coercePresentOptions`
   * - Register a built-in implementation
     - ``edelweissfe/config/registry.py``, the ``_BUILTINS`` table
   * - Register a third-party implementation
     - an ``"edelweissfe.plugins"`` entry point in the plugin's ``pyproject.toml``
   * - Look something up by name at runtime
     - :func:`edelweissfe.config.registry.lookup`
   * - Add a new top-level ``.inp`` keyword
     - :class:`edelweissfe.keywords.base.keywordbase.KeywordBase`
   * - Render a schema's option table in the docs
     - the ``.. pprint:: category:name`` directive (``doc/source/conf.py``)
