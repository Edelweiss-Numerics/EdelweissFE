---
name: ew-add-element
description: >-
  Procedure for adding, registering, testing, and documenting a new finite element formulation in EdelweissFE.
  Use when implementing continuum elements, structural/beam elements, mixed/EAS elements, or Marmot element wrappers.
---

# Implementing a Finite Element in EdelweissFE

## 1. Node Ordering Conventions (Abaqus/Marmot Standard)
- **Quad4**: `1, 2, 3, 4` counter-clockwise
- **Quad8 / CPS8R**: Corners `1..4`, mid-sides `5..8`
- **Hexa8 (C3D8)**: Bottom face `1..4` CCW, top face `5..8` CCW
- **Hexa20 (C3D20)**: Bottom corners `1..4`, top corners `5..8`, bottom mid-sides `9..12`, top mid-sides `13..16`, vertical mid-sides `17..20`

## 2. Base Classes & Reusable Helpers
```python
from edelweissfe.elements.base.displacementelement import (
    DisplacementElement,
)
from edelweissfe.elements.base.displacementtlelement import (
    DisplacementTLElement,
)
```
- **Reuse**: Shared shape functions, Gauss quadrature rules, and $B$-matrix routines in `edelweissfe/elements/` and `edelweissfe/numerics/`.
- **Properties**: Support string-keyed properties via `*elementproperty`.

## 3. Registration
1. **Metadata**: In `edelweissfe/elements/library.py` (node count, DOFs per node, integration points).
2. **Lazy Loader**: In `edelweissfe/config/elementlibrary.py`:
   ```python
   elif strCaseCmp(elementType, "myElement"):
       from edelweissfe.elements.myfamily.myelement import MyElement

       element = MyElement
```

## 4. Tests & Documentation
- **Tests**: Follow [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) using `*modelGenerator`.
- **Docs**: Follow [`ew-documentation`](../ew-documentation/SKILL.md) (update `doc/source/documentation/elements.rst`).
- **Review**: Follow [`ew-code-review`](../ew-code-review/SKILL.md).
