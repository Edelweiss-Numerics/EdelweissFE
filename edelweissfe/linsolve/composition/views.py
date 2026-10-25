#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Views of a composed solver, all drawn from the components' :meth:`~edelweissfe.linsolve.composition.base.
Preconditioner.describe` and the tree of their :meth:`~edelweissfe.linsolve.composition.base.Preconditioner.children`:

- :func:`structure`: the tree as text, for the terminal and the journal (also ``print(solver)``);
- :func:`formula`: the composed approximate inverse, in unicode or LaTeX;
- :func:`dataFlow`: one application as a graphviz graph (DOT text, or SVG if the ``graphviz`` package is installed);
- :func:`runStatistics`: the tree annotated with what each component recorded while it ran;
- :func:`spy`: the sparsity patterns of the matrices the components were built on (matplotlib);
- :func:`html`: all of the above in one self-contained page (also shown inline in Jupyter).
"""

import html as htmlEscaping

#: The format of the parameters of a component in the structure view.
_PARAMETER_FORMAT = "{:}={:}"


def _parameters(description) -> str:
    return ", ".join(_PARAMETER_FORMAT.format(key, value) for key, value in description.parameters.items())


def _tree(component, label, lines: list, prefix: str, isLast: bool, isRoot: bool, annotate) -> None:
    description = component.describe()
    parameters = _parameters(description)
    text = (label + ": " if label else "") + description.name + ("(" + parameters + ")" if parameters else "")
    annotation = annotate(component)
    if annotation:
        text += "   " + annotation
    if isRoot:
        lines.append(text)
        childPrefix = ""
    else:
        lines.append(prefix + ("└─ " if isLast else "├─ ") + text)
        childPrefix = prefix + ("   " if isLast else "│  ")
    children = component.children()
    for index, (role, child) in enumerate(children):
        # a role that only says "the child" is no label worth printing
        childLabel = "" if len(children) == 1 and role in ("inner", "preconditioner") else role
        _tree(child, childLabel, lines, childPrefix, index == len(children) - 1, False, annotate)


def structure(component) -> str:
    """The tree as text: one line per component, its parameters, and the role of each child.

    Example::

        LGMRES(tolerance=ForcingTolerance(minimum=0.0001, maximum=0.001), maxIterations=100, restart=50,
        augmentationVectors=3)
        └─ DiagonalScaling(D=diag(1/√|aᵢᵢ|))
           └─ Multiplicative
              ├─ M1: OnField(field=displacement)
              │  └─ TwoLevel(coarse space=rigid-body modes, prolongation=smoothed, coarse solve=AMG V-cycle)
              └─ M2: OnField(field=nonlocal damage)
                 └─ AMG(tree=default)
    """
    lines = []
    _tree(component, "", lines, "", True, True, lambda _: "")
    return "\n".join(lines)


def _formula(component, latex: bool) -> str:
    description = component.describe()
    text = description.latex if latex and description.latex else description.formula
    for role, child in component.children():
        text = text.replace("{" + role + "}", _formula(child, latex))
    return text


def formula(component, latex: bool = False) -> str:
    """The composed approximate inverse, each child's formula put in the place of its role, e.g.
    ``LGMRES(A, D · (Rᵀ AMG(nonlocal damage)⁻¹ R ∘ Rᵀ TwoLevel(displacement; rigid-body modes)⁻¹ R) · D)⁻¹``."""
    return _formula(component, latex)


def runStatistics(component) -> str:
    """The tree annotated with what each component recorded (inclusive times: a component's contain its
    children's)."""

    def annotate(c):
        s = c.statistics
        parts = []
        if s.setUps:
            parts.append("set up {:}× {:.2f}s".format(s.setUps, s.setUpSeconds))
        if s.updates:
            parts.append("updated {:}× {:.2f}s".format(s.updates, s.updateSeconds))
        if s.applications:
            parts.append("applied {:}× {:.2f}s".format(s.applications, s.applySeconds))
        parts += ["{:} {:}".format(key, value) for key, value in s.sizes.items()]
        return " · ".join(parts)

    lines = []
    _tree(component, "", lines, "", True, True, annotate)
    return "\n".join(lines)


def _dotEscape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dataFlowNodes(component, path: str, lines: list) -> tuple:
    """Add the cluster of ``component`` to ``lines``; return the ids of its entry and exit nodes."""
    description = component.describe()
    cluster = "cluster_" + path
    lines.append('  subgraph "{:}" {{'.format(cluster))
    lines.append('    label="{:}"; style=rounded; color=gray50; fontname="Helvetica";'.format(description.name))
    childOf = dict(component.children())
    steps = description.steps or ()
    previous, entry = None, None
    edges = []
    if not steps and not childOf:
        node = path + "_leaf"
        lines.append(
            '    "{:}" [label="{:}", shape=box, style=filled, fillcolor=gray95];'.format(
                node, _dotEscape(description.formula)
            )
        )
        lines.append("  }")
        return node, node
    for index, step in enumerate(steps):
        node = "{:}_{:}".format(path, index)
        if step.child is not None and step.child in childOf:
            childEntry, childExit = _dataFlowNodes(childOf[step.child], path + "_" + str(index), lines)
            lines.append('    "{:}" [label="{:}", shape=plaintext];'.format(node, _dotEscape(step.operation)))
            edges.append((node, childEntry))
            stepExit = childExit
        else:
            lines.append('    "{:}" [label="{:}", shape=box];'.format(node, _dotEscape(step.operation)))
            stepExit = node
        if previous is not None:
            edges.append((previous, node))
        entry = entry or node
        previous = stepExit
    lines.append("  }")
    for source, target in edges:
        lines.append('  "{:}" -> "{:}";'.format(source, target))
    return entry, previous


def dataFlow(component, svg: bool = False) -> str:
    """One application as a graphviz graph: each component a box around its steps, a step that delegates leading
    into its child's box and back out. DOT text; with ``svg``, rendered by the ``graphviz`` package (raises
    ``ImportError`` without it)."""
    lines = ["digraph composedSolver {", '  rankdir=TB; node [fontname="Helvetica", fontsize=11];']
    entry, exit = _dataFlowNodes(component, "root", lines)
    lines.append('  "r" [shape=circle]; "z" [shape=circle];')
    lines.append('  "r" -> "{:}"; "{:}" -> "z";'.format(entry, exit))
    lines.append("}")
    dot = "\n".join(lines)
    if not svg:
        return dot
    import graphviz

    return graphviz.Source(dot).pipe(format="svg").decode("utf-8")


def _builtMatrices(component, label: str, found: list) -> None:
    """Collect ``(label, scipy matrix)`` of what the components were built on and kept."""
    found += [(label + name, matrix) for name, matrix in component.builtMatrices()]
    for role, child in component.children():
        _builtMatrices(child, label + role + " / ", found)


def spy(component, A=None, markersize: float = 0.05):
    """The sparsity patterns of ``A`` (if given) and of the matrices the components were built on, side by side
    (matplotlib). Returns the figure."""
    import matplotlib.pyplot as plt

    matrices = [("A", A)] if A is not None else []
    _builtMatrices(component, "", matrices)
    figure, axes = plt.subplots(1, max(len(matrices), 1), figsize=(4 * max(len(matrices), 1), 4), squeeze=False)
    for axis, (label, matrix) in zip(axes[0], matrices):
        axis.spy(matrix, markersize=markersize)
        axis.set_title("{:}  ({:}×{:}, nnz {:,})".format(label, *matrix.shape, matrix.nnz), fontsize=9)
    return figure


def html(component, standalone: bool = True) -> str:
    """The structure, formula, run statistics and data flow in one self-contained HTML page (the data flow as SVG if
    the ``graphviz`` package is installed, else as DOT text)."""
    try:
        flow = dataFlow(component, svg=True)
    except (ImportError, OSError, RuntimeError):
        flow = "<pre>" + htmlEscaping.escape(dataFlow(component)) + "</pre>"
    body = (
        "<h3>Structure</h3><pre>{:}</pre>"
        "<h3>Formula</h3><pre>{:}</pre>"
        "<h3>Run statistics</h3><pre>{:}</pre>"
        "<h3>One application</h3>{:}"
    ).format(
        htmlEscaping.escape(structure(component)),
        htmlEscaping.escape(formula(component)),
        htmlEscaping.escape(runStatistics(component)),
        flow,
    )
    if not standalone:
        return "<div>" + body + "</div>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Composed solver</title>"
        "<style>body{font-family:Helvetica,sans-serif;margin:2em;} pre{background:#f6f6f6;padding:1em;}</style>"
        "</head><body>" + body + "</body></html>"
    )


def writeHtml(component, path: str) -> None:
    """Write :func:`html` to ``path``."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(html(component))
