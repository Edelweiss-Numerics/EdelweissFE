---
name: ew-documentation
description: >-
  Procedure for writing, updating, structuring, and building documentation in EdelweissFE.
  Use when adding or modifying Sphinx docs (.rst files), input keyword references, docstrings, or theory manuals.
---

# Sphinx Documentation Guide

All docs live under `doc/source/`, grouped into one directory per sidebar section (see 2.).

## 1. Local Build Command
```bash
sphinx-build ./doc/source/ ./docs -b html
```
CI fails on Sphinx warnings or broken references.

## 2. Subsystem Documentation
One directory per sidebar section (`model/`, `analysis/`, `output/`, `theory/`, `developer/`); what goes where is
described in `CONTRIBUTING.md` (Documenting New Features). A new page is added to the matching `toctree` in
`doc/source/index.rst`.

### Template for Models / Elements
```rst
My Feature Name
---------------

Governing equations:

.. math::
   \boldsymbol{\sigma} = \mathcal{F}(\boldsymbol{\varepsilon}, \dots)

Parameters:
- ``param1``: Description (:math:`\text{unit}`).
- ``param2``: Description.

Input deck snippet:

.. code-block:: edelweiss

   *material, name=myMat, id=myMaterial, provider=edelweiss
   <val1>, <val2>, ...
```

## 3. Docstrings & Keyword Catalog
- **Style**: NumPy-style docstrings (`Parameters`, `Returns`).
- **Keywords**: Add new keywords to `doc/source/keywords.rst`.
- **CLI query**: `edelweissfe --doc=<Module>` and `edelweissfe --keywords`.
