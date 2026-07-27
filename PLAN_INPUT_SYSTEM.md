# EdelweissFE input-language system: diagnosis and redesign plan

Branch: `feat/input-system-registry` (stacked on `feat/amr-hanging-nodes`).

This document is the foundation (P0) for redesigning `edelweissfe/utils/inputlanguage.py` and
`edelweissfe/utils/inputfileparser.py`. It does **not** implement P1-P6 -- see "Scope of this
branch" at the end.

All file:line references below were verified against the current tree on this branch. Where the
line has drifted from the original analysis (mostly due to unrelated AMR-feature insertions
earlier in the same files), the correction is noted explicitly rather than silently updated, so the
drift itself is visible.

## 1. Diagnosis

| # | Finding | Evidence (verified) |
|---|---------|----------------------|
| 1 | `InputLanguage` is a `@singleton` (decorator at `inputlanguage.py:61`, class at `:62`) serving five roles at once: grammar, validator, default-value store, doc source, and runtime config object. Modules register into it as **import side effects**. | `edelweissfe/utils/inputlanguage.py:61-62`; e.g. `edelweissfe/outputmanagers/ensight.py:66-68` runs `inputLanguage[keyword].addModule(module)` at module import time. |
| 2 | Import order is load-bearing: `inputfileparser.py` wraps its per-subsystem imports in **8** `# isort: off` / `# isort: on` blocks (not 7 as originally counted), spanning lines 261-450. `solvers/nonlinearimplicitstatic.py:47` imports `inputLanguage` *from* `stepactions.options` purely to force ordering. | `edelweissfe/utils/inputfileparser.py`: block starts at lines 261, 291, 304, 335, 366, 401, 423, 447 (ends 265, 294, 309, 356, 378, 413, 437, 450). `edelweissfe/solvers/nonlinearimplicitstatic.py:47`: `from edelweissfe.stepactions.options import inputLanguage` (verified exact). |
| 3 | Registration fails silently: the `if keyword in inputLanguage:` guard (e.g. `outputmanagers/ensight.py:66`) no-ops when the top-level keyword doesn't exist yet at import time. `generators/microstructuregenerator.py` is documented (`doc/source/documentation/generators.rst:104`, verified exact) but is never imported during a normal run: the generator import block that would trigger its registration is at `inputfileparser.py:157-169` (**corrected** -- originally cited as :392-398, which is actually the unrelated `*modelGenerator` top-level-keyword definition a few hundred lines later) and does not include it. | `edelweissfe/utils/inputfileparser.py:157-169` (imports `boxgen`, `cubit`, `executepythoncode`, `findclosestnode`, `pipegen`, `planerectquad`, `cuboidlatticegenerator`, `discreterigidbodygenerator`, `surfaceelementgenerator` -- no `microstructuregenerator`). Confirmed live: importing `edelweissfe.utils.inputfileparser` and then checking `inputLanguage["modelGenerator"].getModule("microstructureGenerator")` fails; importing `edelweissfe.generators.microstructuregenerator` directly afterwards registers it correctly (the guard itself is fine -- it's purely an import-order omission). |
| 4 | 13 of the 16 `stepactions/*.py` modules are **un-importable standalone**: `stepactions/dirichlet.py:48-49` (corrected from :47 -- the unguarded lookup is `inputLanguage["step"].getModule(...)`, two lines, at 48 and 49) does an unguarded `inputLanguage["step"]` lookup, raising `Exception: You tried to find a string similar to step in an empty list.` This is why **EdelweissMeshfree reimplements its own `Dirichlet`** rather than reusing EdelweissFE's. | Verified live: `import edelweissfe.stepactions.<module>` standalone fails with the above exception for `bodyforce, changematerialproperty, dirichlet, distributedload, geostatic, indirectcontractioncontrol, indirectcontrol, initializematerial, modelupdate, nodeforces, options, setfield, setinitialconditions` (13 files; `stepactions/base/*` and `__init__.py` excluded). `EdelweissMeshfree/edelweissmeshfree/stepactions/dirichlet.py:38` defines its own `class Dirichlet(DirichletBase)` with a clean typed constructor (`__init__(self, name, nSet, field, values, model, journal, f_t=None)`) instead of importing EdelweissFE's. |
| 5 | The grammar object doubles as the runtime default-value store. **Bug 1 in this finding was already fixed** before this branch's work started (see §2 below) -- the current, corrected code is at `ensight.py:851-856` (corrected from the original :763-765): `module.getKeyword("configuration")["intermediateSaveInterval"].default` / `["overwrite"].default`, read once at `OutputManager.__init__` to seed instance defaults absent a `>>configuration` line. | `edelweissfe/outputmanagers/ensight.py:851-856`. |
| 6 | 9 output managers' `outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs)` take a `moduleOptions` positional parameter; **only ensight actually reads it** (`ensight.py:791-793`, corrected from :707-709). The other 8 (`computetimemonitor`, `conditionalstop`, `monitor`, `fractureenergyintegrator`, `plotalongpath`, `timemonitor`, `meshdatatofile`, `statusfile`) accept but never use it -- a dead parameter existing only to satisfy the uniform call sites at `helpers/inputfilehelpers.py:453-458` and `:480-485` (corrected from :418-427/:445-454). | `grep -n "def outputManagerFactory" edelweissfe/outputmanagers/*.py` -- 9 matches, all with identical signature; only `ensight.py`'s body references `moduleOptions`. |
| 7 | `helpers/inputfilehelpers.py:72-218` (`createFieldOutputFromInputFile`; corrected end line, was cited as :188 -- the function grew with rigid-body/analytical-field support) is the **one** subsystem that already separates parsing (reads `inputfile["fieldOutput"]`) from construction (calls `fieldOutputController.addPerNodeFieldOutput(name=..., nodeField=..., result=..., ...)` with explicit typed kwargs) correctly. It is the template for the target design. | `edelweissfe/helpers/inputfilehelpers.py:72-218`. |
| 8 | There is **no test coverage of any programmatic path**: `_cli/_run_tests_edelweissfe.py:98` (corrected from :89) routes every single test through `parseInputFile`. | `edelweissfe/_cli/_run_tests_edelweissfe.py:44,98`. This branch adds the first pytest suite under `tests/` (see §4, P0). |
| 9 | Docs are pure reflection over the singleton: a custom `pprint` Sphinx directive, class `PrettyPrintDirective` at `doc/source/conf.py:121-258` (verified exact), registered at `doc/source/conf.py:269` (verified exact), consumes module-level `documentation` lists. There are now **50** `.. pprint::` directives across **10** `.rst` files under `doc/source/documentation/` (corrected -- originally 43/9; the doc set grew, e.g. with AMR-related pages). `doc/source/conf.py:38` (comment block from :34, verified) must call `InputLanguage().ensureParserLoaded()` (`inputlanguage.py:69-87`) to force-populate the half-built singleton before Sphinx reads sources. | `doc/source/conf.py:38,121-258,269`; `grep -c pprint:: doc/source/documentation/*.rst` totals 50 across 10 files. |

### Bug found already fixed (not by this branch)

Finding 5 above ("the runtime default store") originally described a bug where
`self.intermediateSaveInterval` read `module.getKeyword("configuration")["overwrite"].default`
instead of its own key. **This is no longer present in the code.** `git blame` on
`edelweissfe/outputmanagers/ensight.py:851-856` shows it was corrected by commit `ae91ae20`
("Matthias Neuner", 2026-07-12), inherited via this branch's parent `feat/amr-hanging-nodes`,
before this analysis/branch began. The behavior described in the original task brief (no
intermediate `.case` saves ever happening absent an explicit `>>configuration` line) is **not**
reproducible on this branch. See §2 for the regression test that pins the correct (now
already-fixed) behavior down going forward.

## 2. Bugs fixed on this branch (P0.3)

1. **ensight.py `intermediateSaveInterval` default** -- already fixed upstream (see above). Added
   `tests/test_ensight_bugfixes.py::test_intermediateSaveInterval_reads_its_own_default_not_overwrites`
   to catch a regression.
2. **`strtobool()` crashes on a real `bool`** (`edelweissfe/utils/misc.py:291-313`): `strtobool`
   calls `.lower()` unconditionally, so a programmatic caller (e.g. an EdelweissMeshfree script)
   passing `overwrite=False`/`transient=True` as a real Python `bool` raised `AttributeError`, not
   just a wrong value. Fixed by adding `asBool(val: bool | str) -> bool` to
   `edelweissfe/utils/misc.py` (passes a real `bool` through unchanged, otherwise delegates to
   `strtobool`) and routing both call sites in
   `OutputManager.updateDefinition` (`edelweissfe/outputmanagers/ensight.py:966`, `:975`) through
   it. `strtobool` itself is untouched (other callers -- `constraints/tie.py`,
   `constraints/nodetodeformablesurfacepenalty.py`, `outputmanagers/meshplot.py`,
   `utils/inputlanguage.py:473` -- depend on its string-only contract). Verified: reverting the fix
   reproduces the crash and fails the new regression test; the fix removes it without changing any
   numeric test result (full `testfiles/edelweiss-only` and `testfiles/marmot` suites re-run
   identically with and without the fix, see PR verification section in the branch's final report).

## 3. Target architecture

"The input file is a serialization of API calls, not an alternative construction path." Four
layers:

- **L1 -- Python constructors are the truth.** Explicit typed keyword args, real defaults in the
  signature, no `moduleOptions` / `datalines` / stringly-typed bools. Already done right in
  `models/femodel.py:60` (`FEModel.__init__(self, dimension: int)`) and
  `utils/fieldoutput.py:511` (`ElementFieldOutput.__init__`, referenced via body statement at
  `:534`), and in EdelweissMeshfree's own `stepactions/dirichlet.py:38`
  (`Dirichlet.__init__(self, name, nSet, field, values, model, journal, f_t=None)`).
- **L2 -- Schema as immutable data owned by the module.** Frozen dataclasses (stdlib, not
  pydantic), living inside the module that defines the concept, never pushed into a global, never
  mutated by another module.
- **L3 -- An explicit lazy registry** with `importlib.metadata` entry-point discovery, replacing
  the 11 inconsistent `config/*.py` registries (`elementlibrary.py`, `materiallibrary.py`,
  `outputmanagers.py`, `generators.py`, `solvers.py`, ...) and the `InputLanguage` singleton. Lets
  external packages (EdelweissMeshfree) register without editing EdelweissFE.
- **L4 -- The input-file front-end is a thin adapter.** Parse -> validate -> resolve name strings
  to objects -> call L1.

Three rules: (a) no import side effects; (b) no cross-module schema mutation; (c)
case-insensitivity and string coercion live in L4 only.

### Illustrative sketch of the target module shape

Using the ensight output manager as a running example (today: `Module`/`InputLanguage` side
effects at import time, `moduleOptions` threaded three call frames deep, `asBool`/`strtobool`
string coercion mixed into L1 business logic):

```python
# edelweissfe/outputmanagers/ensight.py  (L1 + L2, no import side effects)

@dataclass(frozen=True)
class EnsightConfigurationSchema:
    """L2: owned by this module, immutable, never mutated by inputfileparser.py."""
    overwrite: bool = False
    intermediate_save_interval: int = 10
    el_set: str | None = None
    n_set: str | None = None
    transient: bool = True


class OutputManager(OutputManagerBase):
    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        per_node: list[PerNodeJobSpec] = (),
        per_element: list[PerElementJobSpec] = (),
        configuration: EnsightConfigurationSchema = EnsightConfigurationSchema(),
    ):
        """L1: real typed kwargs, real defaults -- importable and constructible standalone,
        no InputLanguage/Module/singleton involved."""
        ...

    def update_definition(
        self, *, create: str, field_output: FieldOutput, transient: bool = True, ...
    ):
        """L1: takes an already-a-bool `transient` -- no asBool()/strtobool() needed here at
        all, because L4 is responsible for ever having a string in the first place."""
        ...


# edelweissfe/config/registry.py  (L3)
register("outputmanager", "ensight", OutputManager, schema=EnsightConfigurationSchema)


# edelweissfe/utils/inputfileparser.py  (L4, thin adapter only)
def _build_ensight_from_definition(definition: dict, model, journal) -> OutputManager:
    cls, schema_cls = registry.lookup("outputmanager", definition["type"])
    configuration = schema_cls(**_coerce_and_casefold(definition.get("configuration", {})))
    return cls(definition["name"], model, journal, configuration=configuration, ...)
```

Note what disappears entirely from L1: `Module(...)`, `InputLanguage()`, `addOptionalKeyword`,
`moduleOptions`, `datalines`, `caseInsensitiveKwargsChecker`, `castKwargsValuesAndAddDefaults`,
`asBool`/`strtobool`. All of that is L4-only, and only runs when an `.inp` file is actually being
parsed.

## 4. Satisfying the hard requirement

**Requirement:** EdelweissMeshfree (pure Python scripts, no `.inp` files) must be able to load
specified EdelweissFE modules both by direct import and by registry name, with no parser
involvement and no global state.

- **Direct import path:** `from edelweissfe.outputmanagers.ensight import OutputManager,
  EnsightConfigurationSchema` then construct with real kwargs. Works today already for the
  handful of subsystems shaped like `FieldOutputController`/`FEModel`/`Section` (demonstrated by
  this branch's `tests/test_programmatic_model_build.py`, see §5); after P2-P4 it works uniformly
  for output managers, step actions, constraints, sections, generators, solvers.
- **Registry-name path:** `edelweissfe.config.registry.lookup("outputmanager", "ensight")` returns
  `(OutputManager, EnsightConfigurationSchema)` without importing `inputfileparser` or touching
  `InputLanguage` at all -- entry points are discovered via `importlib.metadata`, and each module
  registers itself exactly once, at first lookup (lazy), not as a side effect of any particular
  import order. An external package (EdelweissMeshfree, or a future third-party plugin) registers
  its own entry point in its own `pyproject.toml` and is discoverable the same way, with zero edits
  to EdelweissFE.
- Both paths bottom out in the same L1 constructor call -- there is exactly one way to build an
  `OutputManager`, whether the caller is `edelweissfe`'s own `.inp`-driven L4 adapter or an
  EdelweissMeshfree script.

## 5. Phased roadmap

| Phase | Deliverable | Checkpoint |
|-------|-------------|------------|
| **P0** (this branch) | Safety net: (P0.1) a programmatic, parser-free pytest end-to-end test, as far as legitimately reachable, with a precise gap report where it isn't (§6); (P0.2) a golden/snapshot test of the entire current input-language surface; (P0.3) two independently-verifiable standing bugs fixed with regression tests. | `python -m pytest tests/` green; `run_tests_edelweissfe ./testfiles/edelweiss-only/` and `./testfiles/marmot/` show no *new* failures vs. `feat/amr-hanging-nodes`. |
| **P1** | New primitives, additive only, nothing wired in yet: `edelweissfe/utils/schema.py` (frozen-dataclass helpers, coercion utilities factored out of `inputlanguage.py`), `edelweissfe/config/registry.py` (lazy registry + `importlib.metadata` entry-point discovery), `edelweissfe/utils/inputcontext.py`'s `InputContext` (carries model/journal/plotter/fieldOutputController through L4 without a singleton). | New modules importable standalone with zero existing behavior changed; unit tests for registry lazy-loading and entry-point discovery. |
| **P2** | First vertical slice: output managers. Each of the 9 `outputmanagers/*.py` gets an L1 constructor + L2 schema; `moduleOptions` dead parameter removed from the 8 that don't use it; `helpers/inputfilehelpers.py`'s output-manager construction becomes an L4 adapter over the L3 registry. | All 9 output managers importable and constructible standalone (mirroring `tests/test_ensight_bugfixes.py`'s pattern); `testfiles/*` output-manager-touching cases (`OutputManagers`, `ComputeTimeMonitor`, `StatusFile`, `MeshDataToFileFromBoxGen`, ensight-using cases) unchanged. |
| **P3** | Step actions + namespaced step options, deleting all 16 cross-mutation blocks (the `# isort: off` blocks plus the `stepactions.options`-via-`nonlinearimplicitstatic.py` forced-ordering import). Each of the 16 `stepactions/*.py` gets a real constructor (mirroring EdelweissMeshfree's own `Dirichlet`, finding #4); `StepManager`/`StepDefinition`/`StepActionDefinition` become L4 adapters. This directly unblocks P0.1's identified gap (§6). | All 16 stepaction modules importable standalone (no `inputLanguage["step"]` lookup at import time); a P0.1-successor test that drives an actual `Step`/`StepAction`/solver cycle programmatically, no `.inp` file. |
| **P4** | Constraints, sections, analyticalfields, generators, solvers get the same L1/L2/L4 split; fold the 11 `config/*.py` registries into the single L3 registry. | Every `edelweissfe` module with a `documentation` list is importable standalone (this branch's golden test, generalized, becomes the regression gate); `config/*.py` registries reduced to thin `registry.lookup(...)` wrappers or removed. |
| **P5** | Retire the `InputLanguage` singleton; port the Sphinx `pprint` directive to read L2 schemas via the L3 registry instead of the singleton's `Module`/`InputFileKeyword` objects. | `doc/source/conf.py:38`'s `ensureParserLoaded()` call deleted; doc build reads the registry; `inputlanguage.py`'s `@singleton`/`InputLanguage` classes deleted or reduced to a deprecated compatibility shim. |
| **P6** | Share the seam with EdelweissMeshfree: EdelweissMeshfree's own reimplementations (`stepactions/dirichlet.py:38`'s `Dirichlet`, and any others discovered along the way) either import EdelweissFE's L1 classes directly or register through the same L3 registry via its own entry points. | EdelweissMeshfree's `stepactions/`, `constraints/` duplication against EdelweissFE shrinks measurably; both packages' pytest suites pass unchanged. |

## 6. P0.1 outcome: what was reachable programmatically, and the precise gap

**Reached, no `.inp` file, no `parseInputFile`, no `InputLanguage` involvement at all:**
`tests/test_programmatic_model_build.py` builds a single CPE4 element (`Node`,
`getElementClass("CPE4", "edelweiss")`, `getMaterialClass("linearelastic", "edelweiss")`,
`sections.plane.Section`, `FEModel`, `ElementSet`/`NodeSet`), calls
`section.assignSectionPropertiesToElement(element)` and `model.prepareYourself(journal)`, then
solves the single element to equilibrium directly (assembling `K`/`P` via
`element.computeKernels(...)` and eliminating the fixed DOFs with plain `numpy`) and asserts
physically-grounded, non-hard-coded invariants: global force balance (Newton's third law) and
mirror symmetry of the response, not a magic expected number.

**Where it stops, precisely:** driving that same model through the production
`StepManager`/`StepAction`/`NIST` solver stack requires constructing a `StepAction` (e.g.
`edelweissfe.stepactions.dirichlet.StepAction.__init__(self, name, action, jobInfo, model,
fieldOutputController, journal)` at `edelweissfe/stepactions/dirichlet.py:105`). Its body
(`updateStepAction`, `dirichlet.py:141-168`) unconditionally indexes `action["components"]`,
`action["analyticalField"]`, `action["1"]`..`action["6"]`, `action["f(t)")` -- i.e. `action` must
already be a fully-populated, parser-shaped dict with every optional key present (even as `None`).
Hand-building that dict here would mean writing a second, hidden parser matching
`castKwargsValuesAndAddDefaults`'s default-filling behavior -- explicitly out of scope, and exactly
the anti-pattern this whole redesign exists to remove.

**Minimal L1 change that would unblock it** (this is what P3 should implement): give `StepAction`
a real constructor with explicit typed kwargs, e.g.

```python
def __init__(
    self, name: str, nSet: NodeSet, field: str, values: dict[int, float], model: FEModel,
    journal: Journal, analyticalField: AnalyticalField = None, f_t: Callable[[float], float] = None,
):
```

mirroring EdelweissMeshfree's own `edelweissmeshfree/stepactions/dirichlet.py:38`
(`Dirichlet.__init__(self, name, nSet, field, values, model, journal, f_t=None)`) -- which is, not
coincidentally, exactly what EdelweissMeshfree had to invent for itself because this constructor
didn't exist. The current dict-consuming `__init__` would then become a thin L4 adapter,
constructing `values`/`analyticalField`/`f_t` from the parsed `action` dict and delegating to the
real constructor -- the same split `createFieldOutputFromInputFile` already demonstrates for field
outputs.

This is a genuine, reachable wall, documented rather than papered over: no monkeypatching, no
hidden mini-parser, no faked assertions.

## Scope of this branch

This branch implements P0 only (safety net): the golden test, the programmatic-build test (and its
gap report above), and the two independently-verifiable bug fixes. It does **not** create
`schema.py` or `registry.py`, and does not refactor any module's constructor beyond the two bug
fixes -- that is P1 onward.
