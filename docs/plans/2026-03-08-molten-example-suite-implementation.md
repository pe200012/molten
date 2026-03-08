# Molten Example Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three meaningful, stress-oriented example executables for `molten`: MLP forward, 2D FFT heat stepping, and Monte Carlo Bachelier pricing, each with built-in self-checks.

**Architecture:** Keep executable entry points thin and move example logic into testable library modules under `src/Molten/Examples`. Build the suite in TDD order: shared infrastructure first, then deterministic MLP example, then complex-valued FFT support plus heat-stepper, then Monte Carlo RAND example and README integration.

**Tech Stack:** Haskell, Stack, Hspec, `massiv`, `Program`, `ArrayRuntime`, `FftRuntime`, `RandRuntime`, `GHC.Clock`

---

### Task 1: Register design docs and example modules

**Files:**
- Modify: `package.yaml`
- Modify: `molten.cabal`
- Create: `src/Molten/Examples/Common.hs`
- Create: `src/Molten/Examples/MlpForward.hs`
- Create: `src/Molten/Examples/Heat2dFft.hs`
- Create: `src/Molten/Examples/MonteCarloBachelier.hs`

**Step 1: Write the failing test**

Create import-based smoke coverage in:
- `test/Molten/Examples/MlpForwardSpec.hs`
- `test/Molten/Examples/Heat2dFftSpec.hs`
- `test/Molten/Examples/MonteCarloBachelierSpec.hs`
- `test/Spec.hs`

Expected initial failure: modules do not exist.

**Step 2: Run test to verify it fails**

Run:
```bash
stack test --test-arguments '--match "Molten.Examples"'
```

Expected: FAIL with missing module errors.

**Step 3: Write minimal implementation**

Create module skeletons and wire them into `package.yaml` / `molten.cabal` / `test/Spec.hs`.

**Step 4: Run test to verify it passes**

Run the same command and expect module resolution to succeed.

---

### Task 2: Add shared example infrastructure

**Files:**
- Modify: `src/Molten/Examples/Common.hs`
- Test: `test/Molten/Examples/CommonSpec.hs`
- Modify: `test/Spec.hs`

**Step 1: Write the failing test**

Test:
- monotonic timing helper returns non-negative durations
- absolute / relative tolerance helpers accept close values and reject distant values
- simple key-value argument parsing handles `--name value`

**Step 2: Run test to verify it fails**

Run:
```bash
stack test --test-arguments '--match "Molten.Examples.Common"'
```

**Step 3: Write minimal implementation**

Implement:
- `measureOnce`
- `measureRepeated`
- `Approx(..)` / comparison helpers
- tiny `parseFlagMap`
- assertion helpers used by executables and tests

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 3: Add MLP forward example core

**Files:**
- Modify: `src/Molten/Examples/MlpForward.hs`
- Test: `test/Molten/Examples/MlpForwardSpec.hs`

**Step 1: Write the failing test**

Add tests for:
- deterministic host input generation shape
- small `Program` build returns expected output sizes
- GPU result matches CPU reference on a shadow case
- self-check reports failure when tolerance is made artificially too small

**Step 2: Run test to verify it fails**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.MlpForward"'
```

**Step 3: Write minimal implementation**

Implement:
- `MlpConfig`
- host tensor builders
- `buildMlpProgram`
- `runMlpShadowCheck`
- `runMlpStress`
- result summary type

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 4: Add complex `NumericExp` support

**Files:**
- Modify: `src/Molten/Array/Expr.hs`
- Test: `test/Molten/Array/ExprSpec.hs`

**Step 1: Write the failing test**

Add tests for:
- `evaluateBinaryExpression` on `Complex Float` multiply and add
- `renderExpression` for complex add/mul produces valid helper-based HIP code rather than raw `float2 * float2`

**Step 2: Run test to verify it fails**

Run:
```bash
stack test --test-arguments '--match "Molten.Array.Expr"'
```

**Step 3: Write minimal implementation**

Implement:
- `NumericExp (Complex Float)`
- `NumericExp (Complex Double)`
- render helpers for complex add/mul
- any helper functions required by generated kernel source

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 5: Make JIT kernels accept complex arithmetic

**Files:**
- Modify: `src/Molten/Array/Runtime.hs`
- Test: `test/Molten/Array/RuntimeSpec.hs`

**Step 1: Write the failing test**

Add tests for:
- map / zip kernel source for complex signatures includes complex helper definitions
- GPU `zipWithArray` can multiply `Complex Float` arrays elementwise on a small case

**Step 2: Run test to verify it fails**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Array.Runtime"'
```

**Step 3: Write minimal implementation**

Update kernel source generation so complex helper functions/macros are emitted when needed.

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 6: Add 2D FFT heat-stepper example core

**Files:**
- Modify: `src/Molten/Examples/Heat2dFft.hs`
- Test: `test/Molten/Examples/Heat2dFftSpec.hs`

**Step 1: Write the failing test**

Add tests for:
- host multiplier generation shape and finite values
- repeated run on same seed / same initial state is stable
- heat-step result does not increase `L2` energy beyond tolerance on a small case

**Step 2: Run test to verify it fails**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.Heat2dFft"'
```

**Step 3: Write minimal implementation**

Implement:
- `Heat2dConfig`
- host initial field and spectral multiplier builders
- eager GPU stepper using `FftRuntime` + `ArrayRuntime`
- summary and self-check helpers

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 7: Add Monte Carlo Bachelier example core

**Files:**
- Modify: `src/Molten/Examples/MonteCarloBachelier.hs`
- Test: `test/Molten/Examples/MonteCarloBachelierSpec.hs`

**Step 1: Write the failing test**

Add tests for:
- Program output size for payoff statistics
- repeated same-seed runs match on a small case
- estimated price is non-negative and CI contains estimate

**Step 2: Run test to verify it fails**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.MonteCarloBachelier"'
```

**Step 3: Write minimal implementation**

Implement:
- `MonteCarloConfig`
- Bachelier payoff program builder
- repeated-run self-check
- confidence interval summary

**Step 4: Run test to verify it passes**

Run the same command.

---

### Task 8: Add the three executable entry points

**Files:**
- Create: `app/mlp-forward/Main.hs`
- Create: `app/heat2d-fft/Main.hs`
- Create: `app/monte-carlo-bachelier/Main.hs`
- Modify: `package.yaml`
- Modify: `molten.cabal`

**Step 1: Write the failing test**

Verification is build-based here. Add executable stanzas first with missing source files and confirm build fails.

**Step 2: Run build to verify it fails**

Run:
```bash
stack build --test --no-run-tests
```

Expected: missing `Main.hs` / missing imported example functions.

**Step 3: Write minimal implementation**

Each `Main.hs` should:
- parse a small flag map
- print config
- create `Context`
- run the example
- print summary
- fail hard when self-check fails

**Step 4: Run build to verify it passes**

Run the same command.

---

### Task 9: Update README and example documentation

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

No automated doc test. Instead create a checklist:
- examples section exists
- all three executable names documented
- default workload and validation behavior documented

**Step 2: Verify checklist is unmet**

Search README before editing.

**Step 3: Write minimal implementation**

Add a concise `Examples` section with commands and what each example stresses.

**Step 4: Verify checklist is satisfied**

Re-read the edited section.

---

### Task 10: Final verification

**Files:**
- Verify whole tree

**Step 1: Run focused tests**

```bash
stack test --test-arguments '--match "Molten.Examples.Common"'
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.MlpForward"'
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.Heat2dFft"'
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Examples.MonteCarloBachelier"'
```

**Step 2: Run full build and tests**

```bash
stack build
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test
```

**Step 3: Run all three executables**

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-mlp-forward
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-heat2d-fft
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-monte-carlo-bachelier
```

**Step 4: Append Implementation Results**

Record:
- which tests passed
- runtime observations
- any deviations from design
