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


def _updateSchemaFor(category: str, name: str):
    """The ``update<name>`` companion schema, for the three step actions that declare one
    (``dirichlet``, ``distributedload``, ``nodeforces``), or ``None``.

    Reuses ``inputfileparser._updateStepActionSchema`` -- the parser's own lookup for the same
    schema, used to validate a partial re-declaration of an already-defined step action in a
    later step -- rather than re-declaring the ``{name: schema}`` mapping a second time here.
    """
    from edelweissfe.utils.inputfileparser import _updateStepActionSchema

    if category != "stepaction":
        return None
    return _updateStepActionSchema(name)


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
        """Render a list of legacy grammar-declaration objects. Dead in practice since U4 (no
        remaining ``.rst`` source passes a list-shaped dotted path -- every former
        ``documentation = [module]`` list was deleted along with the mechanism that built it) --
        kept only as a defensive fallback for :meth:`run`'s legacy dotted-path branch, whose only
        live callers are dict-shaped.
        """
        result = []
        for item in member_data:
            # Duck-typed: whatever `item` is, only `.name`/`.requiredArgs`/etc, if present, matter.
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

    def _render_scalar_fields_table(self, caption, fields):
        """A single Option/Type-Default/Description table from a ``{optionName: Field}`` mapping
        (as returned by ``optionNames``/``scalarOptionNames``) -- shared by the top-level table and
        every ``>>`` sub-keyword's own table, and by the ``update<name>`` companion schema's table.
        """
        import dataclasses

        from edelweissfe.utils.schema import fieldSchemaMeta

        table, head, body = self._make_table(caption, ncols=3)
        row = nodes.row()
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Type / Default")))
        row += nodes.entry("", nodes.paragraph("", nodes.Text("Description")))
        head += row

        for optionName, field in fields.items():
            meta = fieldSchemaMeta(field)
            if meta.required:
                self._add_literal_row(body, optionName, f"{meta.dtype.__name__} (required)", meta.description)
                continue
            if field.default is not dataclasses.MISSING:
                default = field.default
            elif field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                default = None
            self._add_literal_row(body, optionName, f"{meta.dtype.__name__}, default={default!r}", meta.description)
        return table

    def _render_registry_entry(self, category, name, caption):
        """Render ``category:name`` from the L3 registry and its L2 schema -- the sole source of
        truth for what a target accepts (``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U4). ``optionNames``
        (rather than ``scalarOptionNames``) is used for the top-level table so that a
        ``structuralOnly`` field (one an L4 adapter resolves and pops before the schema is built,
        e.g. a constraint's ``nSet``) still renders -- it is real, documented grammar, not an
        implementation detail.
        """
        from edelweissfe.config import registry
        from edelweissfe.utils.schema import (
            optionNames,
            scalarOptionNames,
            subKeywordFieldNames,
        )

        cls, schema = registry.lookup(category, name)
        itemCaption = caption or name

        if schema is None:
            # No L2 schema at all (e.g. generator:executepythoncode, whose datalines are raw
            # Python source with no flat option mapping to describe): nothing to render here, see
            # the target class's own docstring instead.
            table, head, body = self._make_table(itemCaption, ncols=1)
            row = nodes.row()
            row += nodes.entry("", nodes.paragraph("", nodes.Text("Option")))
            head += row
            self._add_row(body, f"{name} declares no option schema -- see its own class docstring.")
            return [table]

        result = [self._render_scalar_fields_table(itemCaption, optionNames(schema))]

        for optionName, field in subKeywordFieldNames(schema).items():
            from edelweissfe.utils.schema import fieldSchemaMeta

            meta = fieldSchemaMeta(field)
            subCaption = f"Keyword: {optionName}" + (" (required)" if meta.required else "")
            result.append(self._render_scalar_fields_table(subCaption, scalarOptionNames(meta.subSchema)))

        updateSchema = _updateSchemaFor(category, name)
        if updateSchema is not None:
            result.append(self._render_scalar_fields_table(f"Keyword: update{name}", optionNames(updateSchema)))

        return result

    def run(self):
        caption = self.options.get("caption", "")

        # New syntax, ``.. pprint:: category:name`` -- e.g. ``stepaction:dirichlet``,
        # ``solver:NIST``. Renders from the L3 registry and the target's own L2 schema, which is
        # the source of truth for what it accepts; see ``_render_registry_entry``.
        if ":" in self.arguments[0]:
            category, name = self.arguments[0].split(":", 1)
            return self._render_registry_entry(category, name, caption)

        # Legacy syntax, a dotted path to a module-level ``documentation`` dict -- still used by
        # the two keywords that are hand-maintained placeholders with no registry entry/schema of
        # their own at all (plotter's ``>>configurePlots``/``>>exportPlots``), so there is no
        # ``category:name`` to look up in the first place. The list-of-items shape this branch
        # used to also handle (every ``documentation = [module]`` module) no longer exists.
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
