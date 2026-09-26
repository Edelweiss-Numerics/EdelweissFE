---
name: ew-add-material
description: >-
  Step-by-step procedure to implement, register, test, and document a new material model in EdelweissFE.
  Use when creating or refactoring material formulations (elastic, hyperelastic, hypoelastic, plastic, or damage models).
---

# Implementing a Material Model in EdelweissFE

## 1. Imports & Core Utilities
Always inherit from base classes and reuse Voigt/tensor utilities:
```python
from edelweissfe.materials.base.basehypoelasticmaterial import (
    BaseHypoElasticMaterial,
)
from edelweissfe.materials.base.basematerial import BaseMaterial
from edelweissfe.utils.exceptions import CutbackRequest
from edelweissfe.utils.voigtnotation import (
    undoVoigtStrain,
    undoVoigtStress,
    voigtStrain,
    voigtStress,
)
```

## 2. Implementation Skeleton & Minimal State Variables
- **Minimal State Variables Requirement**: Store **only strictly path-dependent historical variables** in `stateVars`. Never store instantaneous or algebraically computable quantities unless explicitly requested by the user. Compute derived quantities on the fly.

Create `edelweissfe/materials/<name>/<name>.py`:
```python
import numpy as np

from edelweissfe.materials.base.basehypoelasticmaterial import (
    BaseHypoElasticMaterial,
)


class MyMaterial(BaseHypoElasticMaterial):
    def __init__(self, materialProperties: np.ndarray):
        self.param1 = float(materialProperties[0])
        self.param2 = float(materialProperties[1])
        # Invariant caching: precompute invariant tensors during init
        self._C0 = ...

    def getNumberOfRequiredStateVars(self) -> int:
        return 0  # strictly minimal path-dependent history variables

    def assignCurrentStateVars(self, currentStateVars: np.ndarray):
        self.stateVarsOld = currentStateVars

    def computeStress(
        self, eps, time, dTime, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Return: (stress_voigt, tangent_C_6x6, newStateVars)
        ...
```

## 3. Lazy Registration
In `edelweissfe/config/materiallibrary.py` (inside `provider="edelweiss"`):
```python
elif strCaseCmp(materialName, "myMaterial"):
    from edelweissfe.materials.mymaterial.mymaterial import MyMaterial

    material = MyMaterial
```

## 4. Input Syntax, Tests & Documentation
- **Input deck**: `*material, name=myMat, id=myMaterial, provider=edelweiss` followed by datalines.
- **Tests**: Follow [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) (`*modelGenerator, generator=boxGen` or `planeRectQuad`).
- **Docs**: Follow [`ew-documentation`](../ew-documentation/SKILL.md) (update `doc/source/documentation/materials.rst`).
- **Review**: Follow [`ew-code-review`](../ew-code-review/SKILL.md).
