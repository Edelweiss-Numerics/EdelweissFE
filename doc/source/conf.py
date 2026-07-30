# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# for execution python code in text

try:
    from StringIO import StringIO  # noqa: F401
except ImportError:
    pass

from importlib import import_module

from docutils import nodes
from docutils.parsers.rst import Directive  # noqa: F401
from pygments.lexer import RegexLexer
from pygments.token import Comment, Keyword, Literal, Name, Operator, Text
from sphinx import addnodes  # noqa: F401
from sphinx.directives.code import (  # noqa: F401
    CodeBlock,
    container_wrapper,
    dedent_lines,
)
from sphinx.highlighting import lexers

# Populate the input file language (all keywords and their registered modules) once, eagerly,
# from this clean import context -- before Sphinx starts reading doc sources and
# autodoc/automodule directives start importing individual edelweissfe modules directly.
#
# PrettyPrintDirective itself no longer needs this: since P5, it renders a ported module's own
# options from its L2 schema via the L3 registry (registry.lookup), not by walking a
# fully-populated InputLanguage singleton, and every remaining InputLanguage-consuming code path
# inside it (_declaredArgsFor, _updateKeywordFor, and the legacy dotted-path branch for the
# handful of keywords -- plotter's >>configurePlots/>>exportPlots, *fieldOutput -- that are not
# one-of-many registry entries at all) calls InputLanguage().ensureParserLoaded() itself, lazily,
# on first use.
#
# What this eager call still guards against is a different, pre-existing hazard, unrelated to
# rendering: a module's own Module/Keyword *registration* is itself conditional on its top-level
# keyword already existing (e.g. dirichlet.py's `if "step" in inputLanguage else []`), so a plain
# `automodule::` importing that module standalone -- before anything has imported
# inputfileparser and populated "step" -- makes that registration silently no-op instead of
# failing loudly. _declaredArgsFor would then find no legacy declaration to recover structural
# args from, and the rendered docs would quietly drop a required option (e.g. dirichlet's own
# `nSet`) rather than error -- exactly the kind of silent regression this port is not supposed to
# introduce. Removing this call would only be safe once every ``automodule::``'d module's own
# registration guard no longer depends on import order, which is a pre-existing L4 fragility this
# phase does not touch.
from edelweissfe.utils.inputlanguage import InputLanguage

InputLanguage().ensureParserLoaded()

project = "EdelweissFE"
copyright = "2022, Matthias Neuner"
author = "Matthias Neuner"
release = "v22.07"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autoclass_content = "class"
autodoc_member_order = "groupwise"
# autodoc_typehints = "both"
# less crowded:
autodoc_typehints = "description"

autoclass_content = "init"

napoleon_use_admonition_for_notes = True
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "./edelweiss_fe_logo.png"
# html_logo = "./logo.png"

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]


class EdelweissFELexer(RegexLexer):
    name = "EdelweissFE lexer"
    expression_in_quotes = r'(["\'])(?:(?=(\\?))\2.)*?\1'
    expression_no_quotes = r"\w+"
    expression_no_quotes = r'[^,\n\'"]+'
    equalsign_with_potential_whitespaces = r"\s*=\s*"
    tokens = {
        "root": [
            (r"\s*\*{2}.*\n", Comment.Singleline),
            (r",", Text),
            (r"\*{1}[^,\n]*", Keyword),
            (expression_in_quotes + r"\s*(?==)", Name.Variable),
            (r"(?<==)\s*" + expression_in_quotes, Literal.String),
            (expression_no_quotes + r"\s*(?==)", Name.Variable),
            (r"(?<==)\s*" + expression_no_quotes, Literal.Number),
            (r"=", Operator.Word),
            (r"[^=,\n]+", Text),
        ],
    }


lexers["edelweiss"] = EdelweissFELexer(startinline=True)
pygments_style = "nord"


#: Maps a schema-bearing registry category to the legacy top-level ``.inp`` keyword its modules
#: register a ``Module`` under -- needed only to recover *structural* args (see
#: ``_declaredArgsFor``), never to render options an L2 schema already describes.
_TOP_LEVEL_KEYWORD_BY_CATEGORY = {
    "generator": "modelGenerator",
    "constraint": "constraint",
    "analyticalfield": "analyticalField",
    "modelmodifier": "modelModifier",
    "outputmanager": "output",
}


def _declaredArgsFor(category: str, name: str):
    """The legacy ``Module``/``Keyword`` declaration's required and optional args for
    ``(category, name)``, or ``([], [])`` if there is none to consult.

    This is the *only* remaining consumer, in documentation generation, of the ``Module``/
    ``Keyword`` grammar objects: not because they still describe what a target accepts (its L2
    schema, via the registry, is the source of truth for that now), but because a schema
    deliberately omits *structural* options -- ones that name an existing model object (a node
    set, an element set, the step action's own ``name``) and are resolved by the L4 adapter
    before the schema is even built (see e.g. ``stepactions/dirichlet.py``'s ``DirichletSchema``
    docstring). Those options were still part of the documented grammar under the old renderer,
    so recovering them here is what keeps the new one from silently dropping them.

    ``stepaction`` is nested one level deeper than every other category: a step action registers
    a ``Keyword`` on *each* step-type ``Module`` (identically on all of them), rather than a
    ``Module`` directly on its own top-level keyword, so it needs ``getKeyword`` instead of
    ``getModule``. ``solver`` has no entry at all: a solver's own grammar lives entirely in its
    schema, validated dynamically against a resolved ``*solver``/``*output`` instance by an
    ``>>options`` block (``stepactions/options.py``), not under any keyword of its own.

    Parameters
    ----------
    category
        The registry category (e.g. ``"stepaction"``, ``"constraint"``).
    name
        The name within that category.

    Returns
    -------
    tuple[list, list]
        The declaration's ``requiredArgs`` and ``optionalArgs`` (``KeywordArg`` objects), empty
        if this category/name has no legacy declaration to consult.
    """
    from edelweissfe.utils.inputlanguage import InputLanguage

    il = InputLanguage()
    il.ensureParserLoaded()

    if category == "stepaction":
        if "step" not in il or not il["step"].modules:
            return [], []
        try:
            kw = il["step"].modules[0].getKeyword(name)
        except ValueError:
            return [], []
        return kw.requiredArgs, kw.optionalArgs

    topLevelKeyword = _TOP_LEVEL_KEYWORD_BY_CATEGORY.get(category)
    if topLevelKeyword is None or topLevelKeyword not in il:
        return [], []
    try:
        module = il[topLevelKeyword].getModule(name)
    except ValueError:
        return [], []
    return module.requiredArgs, module.optionalArgs


def _updateKeywordFor(category: str, name: str):
    """The legacy ``update<name>`` companion ``Keyword``, for the three step actions that
    declare one (``dirichlet``, ``distributedload``, ``nodeforces``), or ``None``.

    There is no schema for the update variant -- the L4 adapter revalidates a partial
    re-declaration against this same-named-but-smaller keyword and applies whichever of the
    *same* schema's fields turned out to be present (``coercePresentOptions``) -- so this is
    rendered straight from the legacy declaration, same as before this port.
    """
    from edelweissfe.utils.inputlanguage import InputLanguage

    il = InputLanguage()
    il.ensureParserLoaded()

    if category != "stepaction" or "step" not in il or not il["step"].modules:
        return None
    try:
        return il["step"].modules[0].getKeyword("update" + name)
    except ValueError:
        return None


class PrettyPrintDirective(CodeBlock):
    has_content = True
    optional_arguments = 1
    required_arguments = 1

    def _make_table(self, caption, ncols=3):
        table = nodes.table(cols=ncols)
        group = nodes.tgroup()
        head = nodes.thead()
        body = nodes.tbody()

        if caption:
            title = nodes.title(text=caption)
            table += title

        table += group
        for _ in range(ncols):
            group += nodes.colspec(colwidth=6)
        group += head
        group += body
        return table, head, body

    def _add_row(self, body, *cell_texts):
        row = nodes.row()
        for text in cell_texts:
            row += nodes.entry("", nodes.paragraph("", nodes.Text(text)))
        body += row

    def _add_literal_row(self, body, literal_text, *rest_texts):
        row = nodes.row()
        row += nodes.entry("", nodes.literal(text=literal_text))
        for text in rest_texts:
            row += nodes.entry("", nodes.paragraph("", nodes.Text(text)))
        body += row

    def _render_dict(self, member_data, caption):
        """Render old-style dict documentation."""
        table, head, body = self._make_table(caption, ncols=2)
        row = nodes.row()
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
        head += row
        for key, val in member_data.items():
            self._add_literal_row(body, key, val)
        return [table]

    def _render_inputlanguage(self, member_data, caption):
        """Render new-style InputLanguage list documentation."""
        result = []
        for item in member_data:
            # Each item is a Module or InputFileKeyword
            item_caption = (
                caption if len(member_data) == 1 else f"{caption} [{item.name}]" if caption else f"[{item.name}]"
            )
            table, head, body = self._make_table(item_caption, ncols=3)
            row = nodes.row()
            row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
            row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
            row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
            head += row

            # Required args
            for arg in getattr(item, "requiredArgs", []):
                self._add_literal_row(body, arg.name, f"{arg.dtype.__name__} (required)", arg.description)

            # Optional args
            for arg in getattr(item, "optionalArgs", []):
                default = getattr(arg, "default", None)
                self._add_literal_row(body, arg.name, f"{arg.dtype.__name__}, default={default!r}", arg.description)

            # Required datalines
            dl = getattr(item, "requiredDatalines", None)
            if dl is not None:
                self._add_literal_row(body, dl.name, f"{dl.dtype} (required)", dl.description)

            # Optional datalines
            dl = getattr(item, "optionalDatalines", None)
            if dl is not None:
                self._add_literal_row(body, dl.name, f"{dl.dtype}, optional", dl.description)

            result.append(table)

            # Nested required keywords
            for kw in getattr(item, "requiredKeywords", []):
                kw_table, kw_head, kw_body = self._make_table(f"Keyword: {kw.name} (required)", ncols=3)
                kw_row = nodes.row()
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
                kw_head += kw_row
                for arg in getattr(kw, "requiredArgs", []):
                    self._add_literal_row(kw_body, arg.name, f"{arg.dtype.__name__} (required)", arg.description)
                for arg in getattr(kw, "optionalArgs", []):
                    default = getattr(arg, "default", None)
                    self._add_literal_row(
                        kw_body, arg.name, f"{arg.dtype.__name__}, default={default!r}", arg.description
                    )
                result.append(kw_table)

            # Nested optional keywords (e.g. step actions inside a step module)
            for kw in getattr(item, "optionalKeywords", []):
                kw_table, kw_head, kw_body = self._make_table(f"Keyword: {kw.name}", ncols=3)
                kw_row = nodes.row()
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
                kw_row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
                kw_head += kw_row
                for arg in getattr(kw, "requiredArgs", []):
                    self._add_literal_row(kw_body, arg.name, f"{arg.dtype.__name__} (required)", arg.description)
                for arg in getattr(kw, "optionalArgs", []):
                    default = getattr(arg, "default", None)
                    self._add_literal_row(
                        kw_body, arg.name, f"{arg.dtype.__name__}, default={default!r}", arg.description
                    )
                result.append(kw_table)

        return result

    def _render_args_table(self, caption, requiredArgs, optionalArgs):
        """A single Option/Type-Default/Description table from plain ``KeywordArg`` lists --
        the legacy shape, reused for both the entirely-unported fallback (``schema is None``)
        and the ``update<keyword>`` companion table, neither of which has a schema to read."""
        table, head, body = self._make_table(caption, ncols=3)
        row = nodes.row()
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
        head += row
        for arg in requiredArgs:
            self._add_literal_row(body, arg.name, f"{arg.dtype.__name__} (required)", arg.description)
        for arg in optionalArgs:
            default = getattr(arg, "default", None)
            self._add_literal_row(body, arg.name, f"{arg.dtype.__name__}, default={default!r}", arg.description)
        return table

    def _render_registry_entry(self, category, name, caption):
        """Render ``category:name`` from the L3 registry and its L2 schema (the source of truth
        for a ported module's options), falling back to the legacy ``Module``/``Keyword``
        declaration for structural args a schema deliberately omits, or entirely for a target
        that declares no schema yet. See ``_declaredArgsFor``'s docstring for why either is
        needed at all.
        """
        import dataclasses

        from edelweissfe.config import registry
        from edelweissfe.utils.schema import (
            fieldSchemaMeta,
            scalarOptionNames,
            subKeywordFieldNames,
        )

        cls, schema = registry.lookup(category, name)
        requiredDeclared, optionalDeclared = _declaredArgsFor(category, name)
        itemCaption = caption or name

        if schema is None:
            # Not yet ported to L2 (e.g. generator:executepythoncode, modelmodifier:hadaptivity):
            # the legacy declaration is still the only description of what this target accepts.
            return [self._render_args_table(itemCaption, requiredDeclared, optionalDeclared)]

        scalarFields = scalarOptionNames(schema)
        structuralRequired = [arg for arg in requiredDeclared if arg.name not in scalarFields]
        structuralOptional = [arg for arg in optionalDeclared if arg.name not in scalarFields]

        table, head, body = self._make_table(itemCaption, ncols=3)
        row = nodes.row()
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
        head += row

        for arg in structuralRequired:
            self._add_literal_row(body, arg.name, f"{arg.dtype.__name__} (required)", arg.description)

        for optionName, field in scalarFields.items():
            meta = fieldSchemaMeta(field)
            if meta.required:
                self._add_literal_row(body, optionName, f"{meta.dtype.__name__} (required)", meta.description)
            else:
                if field.default is not dataclasses.MISSING:
                    default = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    default = field.default_factory()
                else:
                    default = None
                self._add_literal_row(body, optionName, f"{meta.dtype.__name__}, default={default!r}", meta.description)

        for arg in structuralOptional:
            default = getattr(arg, "default", None)
            self._add_literal_row(body, arg.name, f"{arg.dtype.__name__}, default={default!r}", arg.description)

        result = [table]

        for optionName, field in subKeywordFieldNames(schema).items():
            meta = fieldSchemaMeta(field)
            subCaption = f"Keyword: {optionName}" + (" (required)" if meta.required else "")
            subScalarFields = scalarOptionNames(meta.subSchema)
            subTable, subHead, subBody = self._make_table(subCaption, ncols=3)
            subRow = nodes.row()
            subRow += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
            subRow += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
            subRow += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
            subHead += subRow
            for subOptionName, subField in subScalarFields.items():
                subMeta = fieldSchemaMeta(subField)
                if subMeta.required:
                    self._add_literal_row(
                        subBody, subOptionName, f"{subMeta.dtype.__name__} (required)", subMeta.description
                    )
                else:
                    if subField.default is not dataclasses.MISSING:
                        subDefault = subField.default
                    elif subField.default_factory is not dataclasses.MISSING:
                        subDefault = subField.default_factory()
                    else:
                        subDefault = None
                    self._add_literal_row(
                        subBody,
                        subOptionName,
                        f"{subMeta.dtype.__name__}, default={subDefault!r}",
                        subMeta.description,
                    )
            result.append(subTable)

        updateKeyword = _updateKeywordFor(category, name)
        if updateKeyword is not None:
            result.append(
                self._render_args_table(
                    f"Keyword: {updateKeyword.name}", updateKeyword.requiredArgs, updateKeyword.optionalArgs
                )
            )

        return result

    def run(self):
        caption = self.options.get("caption", "")

        # New syntax, ``.. pprint:: category:name`` -- e.g. ``stepaction:dirichlet``,
        # ``solver:NIST``. Renders from the L3 registry and the target's own L2 schema, which is
        # the source of truth for what it accepts; see ``_render_registry_entry``.
        if ":" in self.arguments[0]:
            category, name = self.arguments[0].split(":", 1)
            return self._render_registry_entry(category, name, caption)

        # Legacy syntax, a dotted path to a module-level ``documentation`` list/dict -- still
        # used by the handful of keywords that are not one-of-many registry entries at all
        # (e.g. plotter's ``>>configurePlots``/``>>exportPlots``, the top-level ``*fieldOutput``
        # keyword), so there is no ``category:name`` to look up in the first place.
        from edelweissfe.utils.inputlanguage import InputLanguage

        InputLanguage().ensureParserLoaded()

        module_path, member_name = self.arguments[0].rsplit(".", 1)
        member_data = getattr(import_module(module_path), member_name)

        if isinstance(member_data, dict):
            return self._render_dict(member_data, caption)
        elif isinstance(member_data, list):
            return self._render_inputlanguage(member_data, caption)
        else:
            # Fallback: treat as single item
            return self._render_inputlanguage([member_data], caption)


def doi_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # rendered = nodes.Text(text)
    uri = "http://dx.doi.org/" + text
    ref = nodes.reference(rawtext, text, refuri=uri)
    return [nodes.literal("", "", ref)], []


def setup(app):
    app.add_directive("pprint", PrettyPrintDirective)

    app.add_role("doi", doi_role)
