# EdelweissFE input-language system: diagnosis and redesign plan

Branch: `feat/input-system-registry` (stacked on `feat/amr-hanging-nodes`).

This document is the foundation (P0) for redesigning `edelweissfe/utils/inputlanguage.py` and
`edelweissfe/utils/inputfileparser.py`. As of this branch it also implements P1 (new,
additive-only primitives -- see §7). It does **not** implement P2-P6 -- see "Scope of this
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
| 4 | **RESOLVED** on `feat/step-management-on-amr` (`38fc7af`, cherry-picked in as `757ed3c` "refactor(steps): overhaul step management system", sitting between `feat/amr-hanging-nodes` and this branch). Originally: 13 of the 16 `stepactions/*.py` modules were **un-importable standalone**: `stepactions/dirichlet.py:48-49` did an unguarded `inputLanguage["step"]` lookup, raising `Exception: You tried to find a string similar to step in an empty list.` This is why **EdelweissMeshfree reimplements its own `Dirichlet`** rather than reusing EdelweissFE's. As of this branch, step actions register themselves for all available step types (`stepactions/options.py:51`, `inputLanguage["step"].modules` is populated because step-type modules are now imported before step actions), so the `inputLanguage["step"]` lookup no longer hits an empty list. Re-verified live on this branch (2026-07-27): `import edelweissfe.stepactions.<module>` now succeeds standalone for all 13 of the non-base, non-`__init__.py` modules. Correction to the original count while re-verifying: the true denominator is **13**, not "16" -- `find edelweissfe/stepactions -name '*.py'` gives 20 total (14 top-level incl. `__init__.py`, 6 under `base/` incl. its own `__init__.py`); 13 is what remains once both `__init__.py` files and all of `base/*` are excluded, which is what the original finding's own stated exclusion should have counted to begin with. The underlying gap this finding identified -- no real L1 constructor, only a dict-consuming one requiring a fully-populated parser-shaped `action` (see §6) -- is **not** resolved by this fix; only the standalone-*import* symptom is. P3 (below) still owns giving these modules real constructors. | Verified live on this branch: a loop of `python -c "import edelweissfe.stepactions.$f"` over `bodyforce changematerialproperty dirichlet distributedload geostatic indirectcontractioncontrol indirectcontrol initializematerial modelupdate nodeforces options setfield setinitialconditions` -- 13/13 succeed (only a `PYTHON_GIL` re-enablement warning from the transitive Marmot Cython import, no exception). `edelweissfe/stepactions/options.py:47-62`. `EdelweissMeshfree/edelweissmeshfree/stepactions/dirichlet.py:38` still defines its own `class Dirichlet(DirichletBase)` -- P6 territory, unaffected by this fix. |
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
  the (as of this branch) **13** inconsistent `config/*.py` registries -- `analyticalfields.py`,
  `constraints.py`, `elementlibrary.py`, `generators.py`, `linsolve.py`, `materiallibrary.py`,
  `modelmodifiers.py`, `outputmanagers.py`, `sections.py`, `solvers.py`,
  `statetransferstrategies.py`, `stepactions.py`, `steps.py` -- and the `InputLanguage` singleton.
  Lets external packages (EdelweissMeshfree) register without editing EdelweissFE.

  (Correction: this was originally counted as "11". That count was already stale when this plan
  was written -- `modelmodifiers.py` and `statetransferstrategies.py` had already landed via the
  AMR branch (`25dfd05`, `e82bd7a`) without the count being refreshed -- and `config/steps.py` was
  added afterwards, by `38fc7af`/`757ed3c` (see finding #4's resolution above), making it
  unambiguously stale. 13 is the count actually present in `edelweissfe/config/` today, excluding
  `__init__.py` and the three modules that are not name-keyed registries at all
  (`configurator.py`, `phenomena.py`, `timing.py`) and excluding this branch's own new
  `registry.py`.)
- **L4 -- The input-file front-end is a thin adapter.** Parse -> validate -> resolve name strings
  to objects -> call L1.

Three rules: (a) no import side effects; (b) no cross-module schema mutation; (c) **name**
case-insensitivity is a resolver concern and lives in L3; string-to-type coercion and
case-insensitive *option keys* live in L4 only.

(Rule (c) was originally written as "case-insensitivity and string coercion live in L4 only". It
was amended after P1, for two reasons. First, it was never true of the code it described: of the
13 `config/*.py` registries, **12 already casefold the name at the resolver** -- `analyticalfields`,
`constraints`, `generators`, `modelmodifiers`, `outputmanagers`, `sections`, `stepactions`,
`statetransferstrategies` and `linsolve` via `name.lower()`, `steps` via `.casefold()` -- and
exactly one, `solvers.py`, is case-*sensitive*, doing an exact `solverLibrary[name]` lookup on
CamelCase keys followed by `getattr(solver, name)`, so the name doubles as the class attribute and
`"nist"` fails twice over. Name-casefolding at the resolver has been this codebase's de facto
convention all along; P1's registry made it uniform, and the only behavioral delta is that solver
names became case-insensitive -- strictly more permissive, so no existing `.inp` file changes
meaning. Second, and decisively: the §4 hard requirement is that an external package reach these
modules through `registry.lookup` with **no L4 involved**. If name case-insensitivity lived only in
L4, a registry name would resolve differently depending on the front-end it arrived through --
`.inp` accepting `ensight`/`Ensight` while a direct `lookup()` caller got exactly one spelling.
That is precisely the second-class-citizen behavior the registry exists to remove. What rule (c)
still forbids, and what actually *was* wrong with `ensight.py`, is unchanged: no `asBool`/
`strtobool` string coercion inside L1 business logic.)

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
| **P1** (this branch) | New primitives, additive only, nothing wired in yet: `edelweissfe/utils/schema.py` (frozen-dataclass helpers, coercion utilities factored out of `inputlanguage.py`), `edelweissfe/config/registry.py` (lazy registry + `importlib.metadata` entry-point discovery), `edelweissfe/utils/inputcontext.py`'s `InputContext` (carries model/journal/plotter/fieldOutputController through L4 without a singleton). See §7 for what was actually built, its built-in coverage subset, and the thread-safety strategy. | New modules importable standalone with zero existing behavior changed; unit tests for registry lazy-loading and entry-point discovery -- `python -m pytest tests/` green (52 tests), `run_tests_edelweissfe` suites unchanged. |
| **P2** | First vertical slice: output managers. Each of the 9 `outputmanagers/*.py` gets an L1 constructor + L2 schema; `moduleOptions` dead parameter removed from the 8 that don't use it; `helpers/inputfilehelpers.py`'s output-manager construction becomes an L4 adapter over the L3 registry. **Blocking design decision — SETTLED, implemented in `6c73ef4`.** The schema travels on the class: `edelweissfe/utils/schema.py`'s `OptionSchemaProvider` declares a `schema` class attribute defaulting to `None`, `OutputManagerBase` derives from it, and `lookup` reads it through `schemaOf(target)`, so both dotted-string paths (built-in table *and* third-party entry points) now carry schemas. `schemaOf` dispatches on type rather than probing the attribute — load-bearing, not defensive: every `generator` entry resolves to the module-level *function* `generateModelData`, so a bare `target.schema` would raise `AttributeError` at lookup time; function-target categories report `None` until P4 gives them class-based targets. `register(..., schema=...)` still wins where it is used, since it writes straight into the memo cache. `computetimemonitor` carries the first real schema so the built-in path is asserted against a genuine built-in. Original statement of the problem, retained for context: **how a schema reaches the registry along the dotted-string paths.** `registry.lookup` returns `(target, schema)`, but only `registry.register(..., schema=...)` can ever supply a non-`None` schema -- resolution from a dotted string (both the built-in table *and* entry points) hardcodes `(target, None)` at `edelweissfe/config/registry.py:430`. So as soon as schemas exist, the registry-name path is schema-blind, which breaks the §4 hard requirement for exactly the caller it exists to serve: an external package (EdelweissMeshfree, a third-party plugin) discovered via its own entry point would get its schema silently dropped. Pick a convention -- resolve the schema as a declared attribute on the resolved target (e.g. a `schema` class attribute on the L1 class, which keeps schema and constructor in one place and needs no second registration), or a parallel entry-point group -- and apply it uniformly to built-ins and entry points before wiring the first schema in. Note the project rule against `getattr`: a declared base-class attribute, not attribute probing. **Second deliverable: DONE (`ab599ee`).** `RegistryConflictError` (a `RegistryLookupError` subclass, so existing `except RegistryLookupError` guards catch it) is raised on all *three* ways a name can be double-claimed, not just the two originally specified. (1) `register` compares by identity first -- re-registering the identical `(target, schema)` stays idempotent, which tests rely on -- and otherwise raises, naming incumbent and newcomer. (2) `register` also refuses a name the static built-in table owns, checked against the **table of strings**, so it neither imports the built-in nor depends on whether anything resolved it earlier: collision detection must not itself be import-order dependent, which is the pathology this module exists to remove. This required splitting explicit registrations into their own `_registered` dict, separate from the `_resolved` memo cache, for the same reason. (3) `_entryPointDottedString` now collects *all* matching entry points instead of returning the first from an unordered iteration, and raises when they disagree. **A third hazard, found while implementing and not in the original spec: the mirror image of (2).** Because `lookup` consults built-ins *before* entry points, a plugin claiming a built-in name was silently ignored -- its implementation never loaded, no diagnostic anywhere, the plugin author left guessing. `_auditEntryPoints` reports it. It runs **once per process** from the first cache-missing `lookup`, not per lookup, because `entry_points()` costs ~2.3 ms measured and would otherwise be repaid for every distinct name resolved; the trade-off is that the audit reflects the environment as of the first lookup, while per-name resolution keeps re-querying, so a test patching `entry_points` still resolves correctly and merely does not re-audit. Deliberate replacement is still possible, but only via an explicit `override=True` -- silence was the bug, not replacement. Original statement of the problem, retained for context: **make registration collisions raise instead of silently overwriting.** `register` does a bare `_resolved[key] = (target, schema)` (`edelweissfe/config/registry.py:363`) with no conflict check, and because keys are casefolded (see rule (c) above), a plugin registering `"Foo"` silently replaces a built-in `"foo"` -- the built-in simply stops existing, with no diagnostic anywhere. Casefolding is not what creates this hazard, it only widens it: two entry points claiming the same exact name collide identically, and `_entryPointDottedString` (`:289`) returns the *first* match while iterating an unordered `entry_points()` result, so which one wins is not even deterministic across environments. Raise a `RegistryLookupError` subclass (e.g. `RegistryConflictError`) naming both the incumbent and the newcomer, on `register` and on entry-point resolution alike. Deliberately *not* an error: re-registering the identical object under the same name (idempotent, which tests rely on) -- compare by identity before raising. Cheap to do now, and it must be in place before P4 folds 13 registries' worth of names into one namespace, where a collision becomes far likelier and a silent one far harder to trace. **Three findings from implementing the schema convention, all affecting the remaining vertical slice.** (i) **It is 10 output managers, not 9.** `edelweissfe/outputmanagers/` holds 10 modules and the registry covers all 10; the "9" counted the ones with an `outputManagerFactory`. `meshplot.py` has none — `inputfilehelpers.py:434-453` special-cases it by name (`# new input file parsing not yet implemented for meshplot`), constructing the class directly and then feeding each dataline through `updateDefinition(**kwargs)`. So `meshplot` needs the *most* work of the ten, not the least, and is the one case a uniform adapter cannot absorb as-is. (ii) **A latent `None`-coercion trap the L4 adapter must not walk into.** `coerceValue(None, str)` returns the string `"None"` — truthy — so feeding an explicit `None` through `buildSchemaFromOptions` for an optional `str` option would turn "user said nothing" into a filename literally called `None`. Legacy avoids this by construction: `castKwargsValuesAndAddDefaults` (`utils/misc.py:434-438`) coerces only keys actually *present* and otherwise substitutes `arg.default`, and the option kwargs come from `module.parseDatalines(...)`, which returns `{}` when there are no datalines (verified: `parseDatalines([]) -> ([], {})`). Note the *definition-level* dict is different — it **is** `None`-filled by the parser (`inputfilehelpers.py:421` tests `outputManagerKwargs["name"] is not None`) — so an adapter that passes definition-level keys into a schema will hit this immediately. Either strip `None`-valued keys in the adapter (consistent with `getOptionsOfCategory`'s strip-`None`s convention) or have `buildSchemaFromOptions` treat an explicit `None` as "absent, use the default"; decide it once, in one place, and test it, because the failure mode is a silently wrong value rather than an exception. (iv) **Pre-existing import-order-dependent grammar bug, found while porting, fixed separately in `cfe74f9`.** `edelweissfe/outputmanagers/timemonitor.py:47` declares its `Module` under the name **`"computetimemonitor"`** — a copy-paste artifact. Two different modules are therefore registered on the `output` keyword under the same name, one declaring `export` as `addOptionalArg(..., default=None)` and the other as `addRequiredArg(...)`; `Module.getModule` (`utils/inputlanguage.py:198-201`) returns the *first* casefolded match, so **whether `export` is required for `*output, type=computetimemonitor` depends on module import order**. Meanwhile `*output, type=timemonitor` cannot be used at all, because no module is registered under that name. No testfile or example uses `type=timemonitor`, which is why this has gone unnoticed. Not fixed as part of the port because correcting the name *adds* a `timemonitor` module to the documented grammar and so changes the golden surface — it deserves its own labelled commit with a regenerated golden file, not to be smuggled into a refactor that is otherwise grammar-neutral. **That commit is now in (`cfe74f9`), and correcting the name exposed that the module was not merely undeclared but broken:** having been unreachable for long enough, `finalizeIncrement(self, U, P, **kwargs)`, `finalizeStep(self, U, P)` and `finalizeJob(self, U, P)` all still matched a call convention the solvers and the driver abandoned, so the first increment raised `TypeError`. All three now match `OutputManagerBase`, as every other output manager already did, and a job with `*output, type=timemonitor` was run to confirm it writes a monotone time series. The module docstring, which leaked a bare "Created on ..." author header into the rendered grammar documentation, was replaced at the same time. Golden surface regenerated; the diff is confined to this one module. **Generalizable lesson for P3/P4:** a name nothing can reach is a place where drift accumulates silently, so expect every module the redesign makes reachable for the first time to need a runtime smoke test, not just a schema. (v) **When `ensight` is ported, two registry tests break by design:** `test_builtin_lookup_resolves_without_any_entry_point_metadata` and `test_builtin_lookup_matches_direct_import` both assert `schema is None` for `ensight`/`dirichlet`/`NIST`/`plane`. Update them to assert on the *target* only, rather than deleting the assertions. (vi) **`ensight` is ported (`5112ab8`), and it forced the generic nested-sub-keyword mechanism.** ensight's grammar is not a flat option list but repeatable `>>perNode`/`>>perElement`/`>>configuration` blocks, so `utils/schema.py` gained `subKeywordField(...)` (a field holding an immutable *tuple* of sub-schema instances) and `buildSchemaFromOptions` gained a third argument taking the parser's `moduleOptions`. This is deliberately generic, not ensight-shaped: **~25 modules use `>>` sub-keywords today** (all of `sections/`, thirteen `stepactions/`, `utils/fieldoutput.py`, `modelmodifiers/adaptivity/hadaptivity.py`), so P4 needs precisely this. Two things worth knowing before reusing it: the parser injects its own bookkeeping keys (`inputFile` always, plus `datalines`/`explicitlySetArgs`) into *every* `>>` block, so a block must be passed through `utils/misc.py`'s `withoutParserBookkeeping` before validation or **every** input fails; and sub-keyword names are matched case-insensitively here, whereas ensight's old `moduleOptions.get("perNode")` was exact-case and would have silently dropped a `>>pernode` block (no test input spells them non-canonically, so this is a latent-bug fix with no behavioural delta). Verified byte-for-byte: the binary ensight export for `TensionBarQuad4` is identical to pre-change output. Coverage is strong (142 inputs have an active ensight block) with one gap: 141 pass `overwrite=yes`, and the only one that does not is not named `test.inp`, so the `overwrite=False` timestamped-directory branch is preserved by inspection, not coverage. (vii) **`meshplot` is the last output manager and needs three decisions before it can be ported — it is not mechanical work.** Unlike every other module it declares *no* grammar to mirror (just `module.addOptionalArg("dummyArg", "Mesh plot uses old input file parsing.")`), so porting it means *authoring* a grammar for the first time, which is a change to the documented surface rather than a translation of it. (a) **Its option set is intentionally open.** `c`, `ls`, `plotType`, `marker`, `ms`, `markerfacecolor` are never read by `updateDefinition`; the whole raw dict is handed to `plotter.plotXYData`, whose signature documents `plotOptions` as "additional plot options in matplotlib format". A closed frozen dataclass cannot express "these fields plus arbitrary matplotlib kwargs" without a catch-all field, and a catch-all would re-admit exactly the typos described next, defeating the validation. (b) **Validating would reject 5 option names across 10 currently-*running* test inputs**, each of which is a real latent bug being masked today: `legend` (9 uses, 4 in running `test.inp`s) is silently ignored because the plotter reads `label`, so those curves have no legend at all; `axpSec` (3 uses, 2 running) is a typo for `axSpec` that silently falls back to the default `111`; and `>>perNode`/`>>perElement` written *inside a dataline* (6 uses, 5 running) parse into a key literally named `'>>perNode'` and, because `create` is then absent, make the **entire line a no-op** — the requested contour plots are never produced. Fixing these means editing test inputs, i.e. changing test data. (c) **It is a tagged union, not a flat schema:** `create=` selects among `perNode`/`perElement`/`xyData`/`meshOnly`, each with a different required set (`xyData` needs `x`+`y`; `perNode` needs `fieldOutput`; `meshOnly` needs `warpBy` only when `configuration=deformed`), plus an orthogonal `saveFigure` job. The natural L2 shape is one schema per `create=` value dispatched on the tag — a *third* pattern, distinct from both the flat schema and `subKeywordField`. (viii) **meshplot: decision taken (fix the inputs, strict closed schema), prerequisites landed, port itself not started.** The five silently-ignored option names are corrected in the inputs (`7e7bbb0`) and meshplot now has executing coverage for the first time (`fc17b5e`: `testfiles/edelweiss-only/MeshPlot`, pure-Python, all four `create=` arms + `saveFigure`, verified to genuinely drive the branches by falsification). That coverage immediately caught `np.trapz` being removed in NumPy 2 -- breaking meshplot's `integral=True` path *and* the entire purpose of `fractureenergyintegrator`, which still has no testfile; both now use `np.trapezoid`. **What remains is a genuinely new L2 pattern, the third:** meshplot's jobs arrive as *datalines*, not `>>` blocks, and the arm is selected by the `create=` value (`perNode`/`perElement`/`xyData`/`meshOnly`), with `saveFigure` as an orthogonal presence-flagged arm. Neither the flat schema nor `subKeywordField` fits, and the existing L4 adapter is structurally wrong for it: that adapter constructs **one manager per dataline**, whereas meshplot needs **one manager aggregating all datalines** (this is the "one case a uniform adapter cannot absorb as-is" noted in finding (i)). Suggested shape, so the adapter stays generic rather than special-casing meshplot again: a *tagged-dataline* declaration on the schema class -- a tag option name plus a `tag value -> arm schema` mapping, plus a presence-flag arm mapping -- and an adapter branch that, when a schema declares one, builds the arm schemas from every dataline and passes the resulting heterogeneous tuple to a single constructor call. Note that the `xyData` arm must enumerate the six matplotlib passthrough options `plotXYData` actually consumes (`c`, `ls`, `plotType`, `marker`, `ms`, `markerfacecolor`) as real fields; that finite, documented set is what makes a closed schema possible here at all. Also note the adapter currently substitutes model objects into meshplot's datalines (`fieldOutput`/`elSet`/`nSet` become live objects at `inputfilehelpers.py`); under L1 those stay names and are resolved inside the constructor, as every already-ported manager does. (ix) **P2's vertical slice is complete: all 10 output managers are ported (`287c0fc`), and the transitional branch has retired itself as predicted.** meshplot did *not* get the tagged-dataline primitive sketched in (viii). Since meshplot is a candidate for retirement or overhaul, the framework should not grow a third schema pattern for its shape: instead `utils/schema.py` gained `DatalineAggregatingSchema`, a base declaring a single `fromDatalines(datalines)` classmethod, and the adapter dispatches on it by `issubclass` (type dispatch, like `schemaOf`, not attribute probing). A schema of that kind owns its own dataline interpretation -- meshplot's `create=` arm table and `saveFigure` presence flag live in `meshplot.py`, not in the adapter -- so the entire pattern disappears with the last module that uses it, and nothing generic was built on the assumption that this module shape persists. With the last module ported, `outputManagerSchema is None` became unreachable for built-ins, so the pre-schema `outputManagerFactory` fallback path, `config/outputmanagers.getOutputManagerFactoryByName`, and the by-name meshplot special case were all deleted; an output manager reaching the adapter without a schema now raises a message telling the author to declare one. **Three latent bugs surfaced by closing meshplot's option set**, all of the same family as the five option names fixed in `7e7bbb0`: (a) `c`, `ls`, `marker`, `ms`, `markerfacecolor` and `plotType` are read by `Plotter.plotXYData` out of the job dict, but `updateDefinition` built a *fresh* dict and never copied them across, so all 27 `c=` and 21 `ls=` uses in the repository were silently ignored -- the premise of finding (vii)(a), that these options were deliberately passed through, was therefore false, and the passthrough never happened at all; they are now honored, which changes the appearance of affected figures; (b) an unlabelled `perNode`/`perElement` job defaulted its color bar label to the field output *object*, because the adapter had already substituted it, and now uses its name; (c) `heightRatio` defaulted to `False`, now `0.0`, the same golden-ratio sentinel with a coherent type. A dataline selecting two jobs at once (`create=` plus `saveFigure`) now raises rather than emitting both: legacy allowed it only because each arm tolerated the other's options, which is precisely the tolerance the closed schema removes, and no input does it. **Strictness was verified far beyond what the suites reach**: all 35 meshplot blocks across `testfiles/` and `examples/` were built against the new schemas directly, including the 26 whose tests skip for missing Marmot modules and the files the runner never executes (one of which needed a dead `export=True` removed). (iii) **Suggested transitional marker: `schema is not None` is exactly the "has this module been ported?" signal**, so the L4 site can branch on `registry.lookup(...)`'s second element and needs no hardcoded list of ported modules and no attribute probing — and the branch deletes itself as the last module is converted. | All 10 output managers (see finding (i)) importable and constructible standalone (mirroring `tests/test_ensight_bugfixes.py`'s pattern); `testfiles/*` output-manager-touching cases (`OutputManagers`, `ComputeTimeMonitor`, `StatusFile`, `MeshDataToFileFromBoxGen`, ensight-using cases) unchanged. A schema round-trips through `registry.lookup` for at least one built-in *and* one synthetic entry point (the path that is schema-blind today), proving the chosen convention works for the external caller the registry exists to serve, not just for the `register()` seam. Two collision tests: a differing-target re-registration raises, an identical-target re-registration does not. |
| **P3** | **Shrunk substantially by `feat/step-management-on-amr` (`38fc7af`/`757ed3c`)**, which landed the namespaced step-options mechanism (`registerOptionsArg`/`getOptionsOfCategory`, `edelweissfe/stepactions/options.py:65-120`) *before* P3 was implemented, and finding #4 is already resolved (see the table above) -- so P3 is no longer "build a namespaced options mechanism and give every step action a real constructor from scratch". It is now: (a) **unify the two redundant fixes for the same `>>options`-silently-dead bug** that currently live side by side -- `StepAction.explicitlySetOptions` (`stepactions/options.py:137`, from the AMR branch) and the default-`None`-plus-strip-`None`s design (`registerOptionsArg`/`getOptionsOfCategory`, from `38fc7af`) -- by deleting one (see the comment on `explicitlySetOptions` in `stepactions/options.py` for why both currently exist); (b) redirect `stepActionFactory`/`StepManager`'s dispatch at the L3 registry (`edelweissfe/config/registry.py`, category `"stepaction"`, already covers all 13 built-in step actions as of P1) instead of `config/stepactions.py`'s `importlib.import_module`; (c) give each of the 13 `stepactions/*.py` modules a real L1 constructor (mirroring EdelweissMeshfree's own `Dirichlet`, finding #4), demoting the current dict-consuming `__init__` to a thin L4 adapter -- this directly unblocks P0.1's identified gap (§6); (d) fix the **docs regression** described below. **Correctness invariant that must not silently break:** `getOptionsOfCategory` (`stepactions/options.py:89-120`) is only correct as long as *every* option ever registered via `registerOptionsArg` on the shared `options` step-action keyword defaults to `None` -- `getOptionsOfCategory` strips `None`-valued entries to recover "what the user actually specified", so a future option registered with a non-`None` default would silently leak into every category's result. Verified on this branch: `grep -rn 'getKeyword("options")' edelweissfe/` matches **only** `edelweissfe/stepactions/options.py:86` -- i.e. `options` is registered on exactly one call site today, which is what makes the invariant currently easy to audit by inspection; P3 must preserve that (or replace the mechanism so the invariant no longer needs to hold). **Docs-regression deliverable:** step options currently render in the generated docs as `default = None` for every option (e.g. a solver's `defaultMaxIter` used to document as `10`, `linsolver` as `pardiso`) because `registerOptionsArg` hardcodes `default=None` on the shared keyword (that hardcoding is exactly what makes the strip-`None`s trick work) while the *true* defaults now live only in each solver's own `SolverSpecificOptions`. Fix: give `registerOptionsArg` a documentation-only default parameter that is rendered by the docs but does not participate in the strip-`None`s logic (i.e. does not change the registered keyword arg's actual runtime default of `None`). **STATUS (2026-07-28): (a), (b) and (d) are done and committed; (c) is what remains.** (a)+(d) in `f2c8d021`: `explicitlySetOptions` is the mechanism that went, chosen on evidence rather than taste -- a grep across *both* repos showed it was written on every `updateStepAction` and read by nothing, in EdelweissFE or EdelweissMeshfree, so despite its own docstring calling it the belt-and-braces guard it protected no consumer. Deleting it makes the strip-`None`s invariant load-bearing *alone*, and the invariant note above says only that it is currently "easy to audit by inspection" -- which is not the same as safe, since the failure mode is a call site added years from now. So it was made structural instead: `registerOptionsArg` is documented and asserted as the *only* registrar on the shared keyword (it hardcodes the `None`, and a caller cannot supply a default), pinned by a **source-level** test in `tests/test_stepoptions.py` -- deliberately source-level, because no amount of exercising today's call sites can catch tomorrow's. (d) is satisfied by a `documentedDefault` threaded through `addOptionalArg` onto `OptionalKeywordArg`, distinct from `default` via a module-level sentinel (`None`, `0` and `False` are all legitimate documented defaults, so a plain `None` could not mean "not supplied"); the NIST/NEST registrations moved *below* their classes so each documented value is read straight out of `SolverSpecificOptions` rather than transcribed, leaving one source of truth and nothing to drift. ensight's two options have no default of their own -- not writing them leaves the `>>configuration` value in place -- so they document that in prose instead of a fabricated number. **A second-order consequence worth knowing, since it is a real behavioural change:** the parser injects `inputFile`/`datalines`/`explicitlySetArgs` into every options block, so `getOptionsOfCategory`'s result was *never* empty and `if options:` was unconditionally true at both consumer sites; now that those keys are stripped, an ensight options block that sets nothing no longer short-circuits the `Ensight`-category fallback at `ensight.py:1186`. The arc-length consumer was checked and reads only `arcLengthController`/`stopCondition`, so it is unaffected. (b) in `e3155b19`: both dispatch sites went through the registry, but they needed *different* things from it and the distinction is the point. `stepActionFactory` wants the class, so it calls `lookup`. `StepActionCollection.__getitem__` only *validates* a name, and it is asked about categories the input defines no actions in (a solver checking for `indirectcontrol`), so `lookup` there would import step-action modules the simulation never uses, for names the user never wrote -- hence a new non-importing `registry.isRegistered()`, which consults the same three sources in the same order but compares strings only, with a subprocess test pinning that it imports nothing. `_availableNames` also became public `availableNames` (so the `KeyError` can list alternatives, as `RegistryLookupError` already did) and learned about `register`ed names, which it had been omitting. The replaced checks -- `importlib.import_module` and `importlib.util.find_spec("edelweissfe.stepactions." + name)` -- could only ever see step actions shipped inside this package, so no external package could contribute one: precisely the hard requirement in §4. **FINDING while scoping (c): three declared step-action keywords are dead grammar, and this is the timemonitor lesson repeating exactly as that lesson predicted.** `dirichlet.py:84` declares `updateDirichlet`, `distributedload.py:68` declares `updatedistributedload`, and `nodeforces.py:77` declares `updateNodeforces` -- each a documented keyword with its own argument list, describing a *partial* update (supply `name` plus only the values that change). None of them can be used. The parser keys `moduleOptions` by the `>>` keyword name, so `>>updateDirichlet` is routed to a step action *module* called `updatedirichlet`, which does not exist: verified live, it raises `KeyError: 'updatedirichlet' is not a known step action module`. It failed identically before P3(b) (`importlib.util.find_spec("edelweissfe.stepactions." + "updatedirichlet")` returns `None`), so this is pre-existing and not a regression of the registry swap. **Zero testfiles and zero examples use any of the three**, which is precisely why it has gone unnoticed -- the same shape as `timemonitor`'s shadowed `Module` name. What *does* work, verified live in the same probe, is re-declaring the original keyword with the same `name` in a later step (`>>dirichlet, name=left, nSet=..., field=..., 2=0.5` in step 2 logs `Updating "left"`), which routes through `StepManager`'s `action.name in self.stepActions[action.module]` branch to `updateStepAction`. Note that this working mechanism requires **re-supplying the required args** (`nSet`, `field`), so it is not the partial update the dead keywords advertise -- they describe a real convenience feature that was never wired up, not merely a redundant alias. **Deliberately not resolved as part of (c):** deleting documented grammar versus implementing the advertised partial update is a product decision, and coupling it to this refactor would forfeit the property that (c) is grammar-neutral (golden surface unchanged), which is the main safety net for a 13-module port. The port therefore carries the three declarations across untouched and the L2 schemas cover only the reachable keyword; the decision is the user's, in its own commit, with a regenerated golden either way. Nothing can break in the meantime: an input using one of these keywords crashes today. | All 13 stepaction modules importable standalone (no `inputLanguage["step"]` lookup at import time -- already true as of this branch, see finding #4); a single unified mechanism for step options (one of `explicitlySetOptions`/strip-`None`s deleted); a P0.1-successor test that drives an actual `Step`/`StepAction`/solver cycle programmatically, no `.inp` file; generated docs show real per-solver defaults again, not `default = None` everywhere. |
| **P4** | Constraints, sections, analyticalfields, generators, solvers get the same L1/L2/L4 split; fold the (as of this branch) **13** `config/*.py` registries into the single L3 registry -- including `config/steps.py`, added by `38fc7af`/`757ed3c` after this plan's original "11" count was written (see the correction in §3). Also finish the `element`/`material`/`linsolver` categories the P1 registry deliberately left uncovered (see §7) -- `element`/`material` need a design decision on how `provider` composes with `name`; `linsolver` needs its non-uniform entries (inline lambdas, wrapper objects requiring call-site options) turned into real, dotted-string-addressable objects. | Every `edelweissfe` module with a `documentation` list is importable standalone (this branch's golden test, generalized, becomes the regression gate); `config/*.py` registries reduced to thin `registry.lookup(...)` wrappers or removed; `registry.lookup("element", ...)`/`("material", ...)`/`("linsolver", ...)` work. |
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

## 7. P1 outcome: what was built

Three new, additive-only modules, plus unit tests -- nothing existing was wired in to them (P2
does that):

- `edelweissfe/utils/schema.py`: `schemaField`/`SchemaFieldMeta`/`schemaFields` (frozen-dataclass
  option schemas with per-field description/dtype/required metadata attached via
  `dataclasses.field(metadata=...)`, not a parallel dict), plus `coerceValue`,
  `resolveCaseInsensitiveOptions`, and `buildSchemaFromOptions` -- coercion utilities factored out
  of (not reinvented from) `inputlanguage.py`'s `KeywordArg.getValueFromKwargs` /
  `castKwargsValuesAndAddDefaults`. Parity with the legacy casting was verified directly:
  `tests/test_schema.py::test_coerceValue_matches_legacy_KeywordArg_casting_for_string_input`
  constructs a real `edelweissfe.utils.inputlanguage.KeywordArg` and asserts identical output for a
  representative `int`/`float`/`str`/`bool` sample. The one deliberate divergence is the bug this
  branch already fixed in P0: the legacy path crashes (`AttributeError`) on an already-`bool`
  value because `strtobool()` unconditionally calls `.lower()`; `coerceValue` does not reproduce
  that crash (see `test_coerceValue_deliberately_diverges_from_legacy_for_an_already_bool_value`).
- `edelweissfe/config/registry.py`: `lookup(category, name) -> (target, schema)`, `register(...)`,
  and `RegistryLookupError`. Resolution order: in-process memo cache -> static `_BUILTINS` table
  -> `importlib.metadata` entry points (group `"edelweissfe.plugins"`). Built-in coverage is a
  **genuine subset**, not full coverage dressed up -- see the module's own docstring for the
  authoritative list, reproduced here: covered -- `outputmanager` (10), `section` (3), `constraint`
  (12), `stepaction` (13), `generator` (10), `analyticalfield` (3, `mapped.py` excluded -- it does
  not conform to the `AnalyticalField`/`analyticalFieldFactory` shape at all), `solver` (7), `step`
  (2), `modelmodifier` (1), `statetransferstrategy` (3); **not** covered -- `element`, `material`
  (provider-dispatch semantics not designed for a flat `(category, name)` registry yet -- `name`
  alone is not enough information, `provider` changes the lookup rule entirely), `linsolver`
  (entries are not uniformly "a dotted string to a plain class/callable": some are inline
  `lambda`s with no addressable name, others need a wrapper object constructed from call-site
  options). All left to P4, which now also must absorb `config/steps.py` (§3's corrected count).
  Thread-safety: a single `threading.Lock` (`_lock`) guards double-checked-locking around the
  resolve-and-store step in `lookup()` -- chosen over a lock-free/idempotent design because
  resolving a dotted string executes arbitrary module-level code in the target module (not just a
  pure computation), so two threads racing through an unguarded resolve could interleave real
  side effects, not merely redundant work; see the `_lock` docstring in `registry.py` for the full
  reasoning. Verified by
  `tests/test_registry.py::test_memoized_lookup_is_thread_safe_under_concurrent_access`, which
  forces a genuine cache miss and then hits `lookup()` from 16 threads through a `Barrier` under
  `PYTHON_GIL=0`, asserting identity (`is`) of the returned class across all threads.
  Zero-eager-import was verified empirically, not just asserted, by
  `test_importing_registry_does_not_import_any_builtin_category_module` (a fresh subprocess
  diffing `sys.modules` before/after `import edelweissfe.config.registry`). Synthetic entry-point
  discovery (`test_synthetic_entry_point_is_discoverable`) turned out to be fully achievable
  in-process, with no compromise: `importlib.metadata.EntryPoint(name=, value=, group=)` is a
  public, documented constructor, so a fake third-party registration can be built directly and fed
  in via `unittest.mock.patch.object(registry, "entry_points", return_value=[fakeEntryPoint])` --
  no on-disk package install, temporary distribution directory, or `sys.path` manipulation needed.
- `edelweissfe/utils/inputcontext.py`: `InputContext`, a frozen dataclass with `model`, `journal`,
  `plotter`, `fieldOutputController` fields. No lookup methods, no mutation, `journal` is an
  ordinary field supplied by the caller (never a singleton/global, per this codebase's explicit
  convention).

Verification performed for this phase: `python -m pytest tests/ -v` -- 52 passed (up from the 5
that existed at the end of P0 -- `test_ensight_bugfixes.py` (3), `test_inputlanguage_golden.py`
(1), `test_programmatic_model_build.py` (1)); `pre-commit run --files <every file touched>` -- all hooks green
on the final state of every touched/created file;
`OMP_NUM_THREADS=1 PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/` and
`./testfiles/marmot/` -- identical to the pre-existing baseline (the three known pre-existing
failures -- `edelweiss-only/NodeToDeformableSurfaceContactPullOut`,
`marmot/GCDP`, `marmot/IndirectDisplacementControl` -- and nothing else); the golden test
(`tests/test_inputlanguage_golden.py`) passes unchanged, confirming P1 did not alter the grammar
surface (it could not have -- nothing new is wired in yet).

## 8. P3(c) outcome: all 13 step actions ported, and a recorded finding corrected

**Status: complete.** All 13 `stepactions/*.py` now declare a typed L1 constructor and translate the
input file's shape in the two L4 hooks (`fromStepActionDefinition` /
`updateStepActionFromDefinition`) established for `dirichlet` in `c56f974a`. Nothing takes `jobInfo`
or `fieldOutputController` in L1 any more; sets, surfaces, materials, field outputs and analytical
fields arrive as objects, magnitudes as `np.ndarray`, amplitudes as callables.

Two pieces of shared infrastructure came out of it:

- **`stepactions/base/amplitude.py`** — six modules each compiled `f(t)` with their own copy of
  `sp.lambdify(t, sp.sympify(...), "numpy")`, and the copies had drifted on what an *absent* `f(t)`
  means. Now `amplitudeFromExpression` (returns None for None, so "unset" stays distinguishable) and
  `linearAmplitude` (the named default that `f_t=None` selects) live in one place. Deliberately not
  in `utils/math.py`: that module is imported by the solvers, and this one pulls in sympy.
- **`tests/test_stepaction_option_coverage.py`** — a *source-level* audit of every step action's L4
  option handling against the declared grammar, in the spirit of `test_stepoptions.py`'s registrar
  test. It pins four things: every module declares a keyword of its own name; every module overrides
  `fromStepActionDefinition` (i.e. this phase stays done); each hook reads only options the keyword
  it is validated against declares; and no *new* declared-but-never-read option appears. It exists
  because simulation coverage of these modules is far thinner than the file count suggests —
  **41 of the `testfiles/marmot` cases SKIP** on a Marmot build lacking their material or element, so
  e.g. `geostatic` and `initializematerial` have exactly one executing test each. Both new tests were
  falsified (merge instead of replace in `options`; a `definition["nSet"]` read in `nodeforces`'
  update hook) before being trusted.

### The correction: the three `update<keyword>` keywords are NOT dead grammar

The P3 row above records `updateDirichlet`, `updatedistributedload` and `updateNodeforces` as
"documented keyword[s] ... None of them can be used", concluding they "describe a real convenience
feature that was never wired up". **The conclusion is wrong, and the port depended on knowing that.**
What is true is only that a literal `>>updateDirichlet` *line* is unroutable (the parser keys
`moduleOptions` by keyword name and there is no `updatedirichlet` step action module). But the
keyword is not reached that way: `utils/inputfileparser.py`'s `parseModuleKeywordLine` first
validates a `>>nodeforces` line against the `nodeforces` keyword and, if that fails **and** a step
action of the same `name` was declared earlier, re-validates it against `module.getKeyword("update" +
keyword)`. So the `update*` keywords are the schema for a **partial re-declaration**, which is live,
documented and exercised by two tests that pass on this branch:

- `testfiles/marmot/NodeForces/test.inp` steps 2 and 3: `>>nodeforces, name=cloadTop, 2=-10,
  f(t)=sin(t*2*pi)` — no `nSet`, no `field`.
- `testfiles/marmot/GeoStatic/test.inp` step 2: `>>distributedload, name=dlTop, delta=-10.0,
  f(t)=t` — no `surface`, no `magnitude`, no `type`, and it uses `delta`, which the
  `distributedload` keyword does not declare at all.

The practical consequence is a trap the port had to avoid and the new audit now guards: an update
hook may only read what the **update** keyword declares. Reading `definition["nSet"]` there works for
a full re-declaration and raises `KeyError` for `NodeForces`. The earlier finding stands only as
"a `>>update*` line cannot be written", and the "partial update was never wired up" claim is
withdrawn.

### Which modules can reach their update hook at all

A second, related reachability rule was found and is worth writing down, because it decides whether a
module's update path is live: `helpers/inputfilehelpers.py` names a step action
`definition.pop("name") or definition.get("category")`, falling back to an **auto-generated unique**
`f"{module}-{counter}"`. So a module that declares no `name` arg gets a fresh name — and therefore a
fresh *instance* — for every declaration, and `StepManager` never takes the update branch.
`indirectcontrol`, `modelupdate` and `setinitialconditions` are in that group (their `name` arg is
commented out in the source); `options` escapes it only because it is keyed by `category`.

For `indirectcontrol` this is a live bug, verified by running a two-declaration input and observing
`Creating "indirectcontrol-0"` then `Creating "indirectcontrol-1"`, never `Updating`: its
`updateStepAction`, `currentL0` and the entire `absolute` formulation are inert via the `.inp`
front-end, and because `nonlinearimplicitstaticparallelarclength.py:89` takes
`[... step.actions["indirectcontrol"].values()][0]`, the controller in force is the *first ever
declared* — so `IndirectDisplacementControl2`'s per-step `L` has never taken effect. Not fixed:
declaring a `name` is a grammar change and having the solver honour the option's value is a product
decision. The hooks are implemented anyway, since they are reachable programmatically.

### Bugs fixed, and bugs recorded

Fixed (each could never have worked, so none is a behaviour change for a working input):

- **`changematerialproperty` crashed for every `provider=edelweiss` material, in two places.**
  `hasattr(self.material, "_materialEnergy")` names an attribute this class never sets (it stores
  `self.theMaterial`) — and `hasattr` does not shield the *object* expression, so it raised
  `AttributeError`; and before that, `self.theMaterial["name"]`/`["properties"]` assume the Marmot
  provider's `dict` record, while an edelweiss material is an *instance*
  (`TypeError: 'NeoHookeanWaMaterial' object is not subscriptable`). The provider-dependent parts are
  now four small `isinstance`-dispatching helpers; the Marmot path is unchanged (log text compared
  verbatim). Proven with a throwaway edelweiss-provider input: the step action changes the result
  (`RF` 2807.07 without it, 3447.38 with it), so the rebuilt material really reaches the elements.
- **The `setEnergyFunction` carry-over it was trying to do is provably redundant**, so it was deleted
  rather than repaired: `_materialEnergy` is assigned only by `setEnergyFunction`, which is called
  only from each autodiff material's own `__init__` with `materialProperties["psi_e"]` (grepped
  across both repos), so `type(material)(properties)` reinstalls the identical function.
  `sections/base/sectionbase.py:88` has the same redundant call and the same `hasattr`; it is left
  alone, as its own change. (An earlier attempt reached the branch via a `@runtime_checkable
  Protocol` — dropped: it is `hasattr` in disguise, and a new idiom in this repo, to reach code that
  does nothing.)
- **`changematerialproperty` without `f(t)`** left `self.f_t` unset and died on the first increment
  with a bare `AttributeError`. `f_t` is now a required constructor argument and the L4 hook raises a
  `ValueError` naming the action. The keyword stays `addOptionalArg` (grammar-neutral).
- **`indirectcontrol` built and discarded a whole `NISTPArcLength` solver instance** on every
  construction *and* every update, storing it in a `self.arcLengthController` that nothing ever read
  (the solver sets its own attribute from the step action *object*). Deleting it removed the module's
  last use of `jobInfo` and the `edelweissfe.stepactions` → `edelweissfe.solvers` import edge — a
  latent cycle, since that solver imports `stepactions.options` at module level. Verified: a fresh
  import of `indirectcontrol` now pulls in no `edelweissfe.solvers` module at all.
- **`np.trapz`**-style silent-drop family, one more instance: `nodeforces` gained a
  `len(nodeForces) != fieldSize` guard, where a mismatch previously crashed later inside
  `PExt[...] += load.flatten()`.

Recorded, deliberately not fixed (all pre-existing; each is a product decision or a reference-solution
change):

- **`indirectcontractioncontrol` is unusable as an arc-length controller**, confirmed by running it:
  the solver checks only that the `arcLengthController` option is *truthy* and then hardcodes
  `step.actions["indirectcontrol"]`, so selecting it raises
  `KeyError: 'Arc length controller "indirectcontractioncontrol" not found in current step'` — or, if
  an `indirectcontrol` action happens to exist, silently uses that instead. It has **zero** testfile
  and example coverage, which is why nobody noticed; per the `timemonitor` lesson it was smoke-run
  and its `cVector` checked programmatically (inward unit vectors scaled by `1/nNodes`).
- **`distributedload`'s `field` option is declared and read by nothing** (`DistributedLoadBase` has no
  notion of a field). Two running inputs write `field=displacement`, which agrees with the unused
  default, so nothing looks wrong. Pinned in the audit's `KNOWN_UNREAD_OPTIONS`.
- **`bodyforce`'s `delta` is unreachable** — it declares no `updatebodyforce` keyword, so a partial
  re-declaration fails to parse and a full one always carries the required `forceVector`, which wins.
  And it could not work if reached: the declared default is the **int** `0` despite `dtype=str`, and
  `np.fromstring(0, ...)` raises `TypeError`. Also pinned.
- **`indirectcontrol`'s `exportCVector` is declared and read by nothing** — only its sibling
  implements it.
- **`absolute` is silently ignored on re-declaration** in both indirect controllers.
- **The three load modules disagree on what a re-declared magnitude means, with identical syntax:**
  `bodyforce`/`distributedload` read it as a new *total* (`delta = total - accumulated`),
  `nodeforces` as an *increment* on top of the accumulated force. Changing either would move
  reference solutions.
- `bodyforce`'s force-vector dimension check runs only in `__init__`, so a later step can install a
  wrong-dimension vector unchecked.

### The `options` container

Ported last and separately, because it is the one step action whose whole purpose is parser
bookkeeping. `__init__(name, category, options)` now takes the options that were actually *set*, and
the strip (parser bookkeeping keys, the `category` tag, and the `None` defaults every foreign module
contributes to this shared keyword) moved from `getOptionsOfCategory` — a function the *solvers* call
— into the L4 seam. So nothing downstream of the constructor mentions the input file, and a
programmatic caller no longer has to hand over a `None`-filled mapping to be stripped again later.

One behaviour that stopped being implied by the mechanism had to be made explicit: `updateStepAction`
**replaces** rather than merges. Pre-port `self.options.update(parsedBlock)` read like a merge but
was a replacement, because the parsed block carries *every* declared arg, so an option omitted in a
later step arrived as `None` and was then dropped at read time. Merging the now-sparse mapping would
silently leak an earlier step's options forward. No simulation test covers this — the only input that
re-declares one category with differing option sets, `testfiles/marmot/IndirectDisplacementControl2`,
skips for a missing Marmot material — hence the dedicated, falsified unit test. The four dead
`__contains__`/`__getitem__`/`__setitem__`/`get` wrappers went too: every consumer reaches these
options through `getOptionsOfCategory`, in both repos.

## Scope of this branch

This branch implements **P0 and P1**. P0 (safety net): the golden test, the programmatic-build
test (and its gap report above), and the two independently-verifiable bug fixes. P1 (new,
additive-only primitives, see §7 for the full account): `edelweissfe/utils/schema.py`,
`edelweissfe/config/registry.py`, `edelweissfe/utils/inputcontext.py`, their unit tests
(`tests/test_schema.py`, `tests/test_registry.py`, `tests/test_inputcontext.py`), the `asBool`
annotation fix (`bool or str` -> `bool | str`) in `edelweissfe/utils/misc.py`, and this revision of
the plan itself to record the `feat/step-management-on-amr` reconciliation (finding #4's
resolution, the corrected `config/*.py` registry count, and P3's shrunken scope). It does **not**
refactor any existing module's constructor beyond the two P0 bug fixes, and does not wire
`schema.py`/`registry.py`/`InputContext` into any existing code path -- that is P2 onward.
