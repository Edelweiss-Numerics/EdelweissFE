---
name: ew-create-regression-test
description: >-
  Step-by-step instructions for constructing, validating, and registering regression test decks (test.inp + U.ref) in EdelweissFE.
  Use when adding new test cases for bug fixes, new features, or solver benchmarking.
---

# Creating Regression Tests in EdelweissFE

Tests are full FE input-deck regression tests (`test.inp` + `U.ref`).
- Pure Python/Cython: `testfiles/edelweiss-only/<TestName>/`
- With Marmot: `testfiles/marmot/<TestName>/`

## 1. Generators (`*modelGenerator`)
Use procedural generators (coarse meshes `1..4` elements for fast execution):
- **2D**: `planeRectQuad` (`CPS4`, `CPE4`). Sets: `bottom`, `top`, `left`, `right`, `all`.
- **3D**: `boxGen` (`C3D8`, `C3D20`). Sets: `<name>_bottom`, `<name>_top`, `<name>_left`, `<name>_right`, `<name>_front`, `<name>_back`, `<name>_all`.

## 2. Test Deck Templates

### 2D Template (`planeRectQuad`)
```
*material, name=mat1, id=myMaterial, provider=edelweiss
<param1>, <param2>, ...

*section, name=sec1, thickness=1.0, material=mat1, type=plane
all

*job, name=test2d, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=10.0, y0=0, h=10.0, elType=CPS4, elProvider=edelweiss, nX=2, nY=2

*step, solver=theSolver
maxInc=0.5, minInc=1e-5, maxNumInc=10, maxIter=25
>>dirichlet, name=fix_y, nSet=bottom, field=displacement, 2=0.0
>>dirichlet, name=fix_x, nSet=bottom, field=displacement, 1=0.0
>>dirichlet, name=pull,  nSet=top,    field=displacement, 2=0.05
```

### 3D Template (`boxGen`)
```
*material, name=mat1, id=myMaterial, provider=edelweiss
<param1>, <param2>, ...

*section, name=sec1, material=mat1, type=solid
all

*job, name=test3d, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*modelGenerator, generator=boxGen, name=gen
x0=0, lX=10.0, y0=0, lY=10.0, z0=0, lZ=10.0, elType=C3D8, elProvider=edelweiss, nX=2, nY=2, nZ=2

*step, solver=theSolver
maxInc=0.5, minInc=1e-5, maxNumInc=10, maxIter=25
>>dirichlet, name=fix_z, nSet=gen_bottom, field=displacement, 3=0.0
>>dirichlet, name=fix_y, nSet=gen_bottom, field=displacement, 2=0.0
>>dirichlet, name=fix_x, nSet=gen_bottom, field=displacement, 1=0.0
>>dirichlet, name=pull,  nSet=gen_top,    field=displacement, 3=0.05
```

## 3. CLI Commands
```bash
# 1. Generate reference solution:
run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests <TestName> --create

# 2. Verify test passes (< 1e-6 residual):
run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests <TestName>

# 3. Run full regression suite:
run_tests_edelweissfe ./testfiles/edelweiss-only/
```
