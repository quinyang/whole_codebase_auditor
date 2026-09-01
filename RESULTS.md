# Results

Falcon3-Mamba-7B-Instruct, 4-bit, Tesla T4 (16 GB), 4,000-token context budget.
Corpus: 10 generated repositories, 40 planted cross-file vulnerabilities, 30
audits (near / far / clean per repo). Seeded and reproducible.

## Headline

| metric | value |
|---|---|
| **precision** | **43.8%** (7 TP / 16 reported) |
| **recall** | **17.5%** (7 / 40 planted) |
| F1 | 0.250 |
| repos with ≥1 detection | 5 / 10 |

Only findings **grounded** to a verified `file:line` are scored. Ungrounded
findings are excluded from both numerator and denominator.

## Ablation: does packing proximity matter?

| separation | recall |
|---|---|
| near (tightly coupled files) | 15.0% (3/20) |
| far (distant files) | 20.0% (4/20) |
| **delta** | **−5.0%** |

**Null result.** Graph-ordered packing shows no measurable recall benefit at this
scale, and the sign is slightly negative. With 20 planted vulnerabilities per arm
this is well inside noise — the honest reading is "no detectable effect", not
"proximity hurts".

This is worth reporting *because* it is negative. The packer's ordering was
motivated by a real argument (an SSM compresses history into a fixed-size state,
so distance should cost recall), and the measurement does not support it at 4k
context on 10-file repositories. It may well matter at 32k on a 500-file repo —
which is exactly the regime a T4 cannot reach.

## The correction that matters most

**Session 2 claimed grounding was a precision filter. That claim does not
survive a larger sample.**

| | proposed | grounded | grounding rate |
|---|---|---|---|
| injected repos | 56 | 10 | **17.9%** |
| clean repos (controls) | 34 | 7 | **20.6%** |

Grounding rejects roughly four out of five proposals *regardless of whether the
repository contains a planted vulnerability*. On the clean controls it let
through 7 findings, every one of which is a false positive — they account for
**all** of the false positives in the headline precision figure.

Session 2 measured 100% grounding on one planted repo and 0% on one clean repo
and concluded the manifest was doing precision work. With n = 2 that separation
was indistinguishable from chance. At n = 30 it disappears.

What grounding *does* do is unchanged and still worth stating: it guarantees
every reported finding points at a real line of real source, so a reviewer can
check it in seconds. That is a usability and auditability property. It is not a
correctness filter, and the earlier framing overstated it.

## Packer coverage (measured separately)

At whole-repo scale and a 4,000-token budget, the packer surfaces **72%** of
planted vulnerabilities with both halves in context. The benchmark slices each
repo so coverage is 100% before auditing, so detection and retrieval are measured
independently rather than confounded.

## Scope and limitations

- **Synthetic corpus.** Ten generated repositories of 9–13 modules, varied by
  layout, dependency shape and domain. Real libraries were attempted first and
  rejected by measurement: a single 350-line module is ~5,600 tokens, and the
  T4's ceiling is ~5,300 tokens total, so both halves of a cross-file defect
  cannot be in context. The synthetic corpus is a consequence of the hardware
  finding, not a shortcut around it.
- **Two vulnerability patterns** (credential leak, taint-to-sink). Injected code
  is recognisably synthetic; a model may be responding to its shape rather than
  its semantics.
- **n = 40 planted, n = 10 repos.** Confidence intervals on 17.5% recall are
  wide. Treat one significant figure as the honest precision of the estimate.
- **Grounding filters unverifiable findings, not wrong ones.** A model can quote
  a real line and still misjudge it — 7 of them did.
- **Single model, single budget, single GPU.** No claim about other SSMs, other
  context lengths, or hardware with fused kernels available.

## What this supports saying

> A 7B pure-SSM auditor detects 17.5% of planted cross-file vulnerabilities at
> 43.8% precision on a purpose-built 10-repository benchmark, with every reported
> finding grounded to a verified file and line. A near/far ablation found no
> measurable effect of graph-ordered context packing at this scale.

## What this does not support saying

- That the tool is useful for real security work. It is not, at this recall.
- That grounding filters hallucinations. Measured: it does not.
- That graph ordering improves detection. Measured: no effect found.
- Anything about real repositories. None were audited.
