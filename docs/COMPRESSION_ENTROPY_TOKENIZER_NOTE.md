# Compression/Entropy Note For The Tokenizer

Source:

- 3Blue1Brown, "Reinventing Entropy | Compression is Intelligence Part 1": https://www.youtube.com/watch?v=l6DKRf-fAAM
- Grant Sanderson's companion note: https://3blue1brown.substack.com/p/reinventing-entropy
- SCBE multipath band source for the three trit classes:
  `C:\dev\SCBE-AETHERMOORE\src\symphonic\multipath\band.py`

## Applied Decision

The Six Tongues tokenizer is a reversible byte-to-token layer, not a compression
algorithm. Each tongue preserves exactly one byte per token, and cross-tongue
translation preserves the same underlying byte stream.

That means a prettier or more meaningful token surface is not evidence of lower
information content. Compression claims need a corpus measurement.

## What Was Added

`aethermoore.py entropy` reports:

- input byte count
- unique byte count
- byte entropy in bits per byte
- Shannon lower bound in bytes for the observed byte distribution
- raw zlib size
- tongue token surface size
- zlib size of the token surface

Use it as the cheap guardrail before claiming that a tongue layout is more
compact, more efficient, or more model-friendly.

`aethermoore.py predict-compress` now tests the prediction claim directly:

- traditional causal next-byte predictor
- trit light-class predictor over six base-3 symbols per input byte
- both scored as cross-entropy / ideal compression bits
- both normalized as `bits/input_byte`
- winner and delta reported honestly

This is the fair local check against a traditional next-token method. It is not
a trained neural net; it is the compression-equivalent loss measurement that a
next-token model would be trying to lower.

`aethermoore.py trit-overlay` turns the binary prefix table into a ternary table
overlay. The three trit classes are mapped to the SCBE multipath band classes:

- `0 -> blocker(-1)`
- `1 -> reader(0)`
- `2 -> writer(+1)`

The overlay reports ternary interval position, ideal binary cost
`depth * log2(3)`, and which fixed-depth binary prefix cells are touched. No
image renderer is part of this path.

## Projection Depth Model

The projection idea is not a globally self-similar fractal and it is not a
rendering task. It is a causal relationship model with local self-similar
overlays inside a deliberately non-self-similar global stack.

Working name: **quasi-metronomic quadratic stenographic array**.

- `quasi`: random distortion plus irrational depth phase makes the whole
  structure aperiodic
- `metronomic`: rationally spaced local rays provide the repeating rhythm
- `quadratic`: all `n(n-1)/2` context-point relationships are measured, with
  an optional second-order product term
- `stenographic`: cross-entropy measures the shortest predictive shorthand
- `array`: rays, depth layers, and backdrop values remain addressable

For an eight-trit history `H_t`, trit `s_j` has vote embedding
`v(s_j) in {-1, 0, +1}`. At depth `d`, a seeded fixed distortion field moves
each history point:

```text
z[j,d] = (position[j] + epsilon_x[j,d],
          v(s_j)      + epsilon_y[j,d])
```

Each local projection array has four rationally separated rays. Successive
depths receive an irrational phase shift:

```text
theta[d,r] = 2*pi * (r/4 + d/phi)
```

Every ray overlays the same dyadic scale family `1, 1/2, 1/4`. The response is
the aggregate of the relation-line projections at those scales. For each
temporally ordered pair `i < j`, the model measures its projected slope:

```text
line[d,r,i,j] = (project(z[j,d]) - project(z[i,d])) / (j - i)
pair[d,r,i,j] = project(z[i,d]) * project(z[j,d]) / sqrt(j - i)
```

The ray response combines the projected-point mean with the mean of all 28
pairwise lines; `--quadratic-gain` controls the optional pair-product term.
Before comparing depths, the backdrop rotates every prior ray into the current
ray direction with a first-harmonic reconstruction `A`:

```text
A(P[k], theta[d,r]) = (2 / rays)
                       * sum_q(cos(theta[d,r] - theta[k,q]) * response[k,q])

backdrop[d,r] = A(P[d-1], theta[d,r])
                + layer_gain * backdrop[d-1,r]
```

This phase compensation is the "self-aligning" operation; comparing equal ray
indices directly would mistake the intentional irrational depth rotation for
non-self. Depth zero reconstructs from its current ray array. Two channels are
retained:

```text
signal    = tanh(projected relation response)
alignment = 1 - abs(tanh(response - backdrop))
```

`alignment` is the self/non-self quantity: agreement with the deep backdrop is
near one; divergence is near zero. The 24 projected values run as a parallel
branch beside the original 24-value one-hot trit context. The flat control uses
a duplicate flat branch. This lets the shared output head ignore a harmful
projection instead of destroying base context through a forced sum.

Every model therefore exposes 48 features to the same 147-parameter softmax.
`--projection-gain` scales the projected branch, `--layer-gain` controls the
recurrent backdrop, `--quadratic-gain` adds the pair-product term, and
`--reinforcement-gain` applies `signal *= 1 + gain * alignment`. Each setting is
identical in aligned and scrambled arms.

The resulting feature array feeds a three-class softmax:

```text
p(s_t | H_t) = softmax(W * features(H_t) + b)
loss_bits    = -log2 p(actual s_t | H_t)
```

Because blocker/reader/writer are three mutually exclusive outcomes, the
compression-correct loss is categorical cross-entropy. Binary bits are the
measurement unit. Independent binary cross-entropy heads would not form one
normalized ternary code without an additional coupling rule.

## Controlled Test

`aethermoore.py projection-benchmark` performs an 80/20 causal split and trains
three models:

- `traditional_flat_context`: one-hot next-trit context duplicated across both
  feature branches
- `scrambled_projection_control`: same projection budget with independent
  random angles and scales
- `aperiodic_relational_projection`: rational local rays, irrational layers,
  dyadic overlays, and the projection backdrop

Each model has 48 features and the same 147 trainable softmax parameters. The
optimizer order is held fixed so the three projection seeds measure geometry,
not training-order noise. The claim gate requires the proposed model to beat
both comparators by more than `2 * pooled_sd`. The command reports `SUPPORTED`,
`UNDERPOWERED`, or `NO_LIFT`.

With `--ablations`, the same run adds:

- `quasi_random_projection`: one random local ray array repeated under the
  irrational golden-ratio depth rotation, with random scales
- `dyadic_scale_projection`: the scrambled-angle control with only its random
  scales replaced by dyadic scales
- `rational_angle_projection`: rational local rays and irrational depth
  rotation, with random scales
- `periodic_rational_dyadic_projection`: the original rational/dyadic geometry
  without irrational depth rotation

The one-factor comparisons are:

- quasi-random versus fully scrambled for structured depth rotation
- dyadic-scale versus scrambled under the same independent random angles
- quasi-random versus rational-angle under the same irrational depth rotation
  and random scales
- original aligned versus rational-angle for dyadic versus random scales
- original aligned versus periodic rational/dyadic for irrational versus
  periodic depth

The quasi-random model also receives its own claim verdict against both the flat
and scrambled controls. A directional gain below the variability gate remains
`UNDERPOWERED`; it is not promoted to evidence by naming it a synthesis.

No hologram, ray-traced image, physical antenna, or perpetual-motion claim is
part of this implementation. "Rays" are rows in a mathematical feature map.

## Current Evidence (2026-07-26)

Using the default 6,000 training trits, 1,500 held-out trits, eight epochs,
projection/layer gains of `1`, quadratic/reinforcement gains of `0`, and seeds
7/20/42. Corpora are exact raw blobs from commit `312b74a60b9`; no text shell
pipeline is allowed to add or remove a newline:

| Pinned corpus | SHA-256 prefix | Flat | Scrambled | Quasi-random | Dyadic | Rational | Periodic | Original |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `README.md` | `6b99970e8ba9` | **7.597437** | 7.610623 | 7.605706 | 7.612059 | 7.617714 | 7.622932 | 7.617258 |
| `aethermoore.py` | `a844a92ae8b0` | 6.485021 | 6.464719 | **6.460641** | 6.466829 | 6.473131 | 6.475824 | 6.474703 |
| `docs/GEOMETRIC_REASONING_FRAMEWORK.md` | `5c1ba105a8a0` | 7.374816 | 7.344673 | **7.336792** | 7.347804 | 7.353313 | 7.357044 | 7.353146 |

Scores are mean held-out ideal bits per input byte; lower is better. Bold marks
the lowest score in each row, not a statistical claim. The quasi-random model
beats fully scrambled projection on all three corpora by `0.004917`, `0.004077`,
and `0.007881` bits/input byte, respectively. None clears the `2 * pooled_sd`
gate, so the direction is repeatable but still `UNDERPOWERED`.

The irrational depth rotation beats the matched periodic rational/dyadic model
by `0.005674`, `0.001121`, and `0.003898` bits/input byte. It clears the gate on
the first and third corpora. Random local rays beat matched rational local rays
by `0.012009`, `0.012490`, and `0.016521`, clearing the gate on all three.
Dyadic scales do not show a repeatable benefit. The supported component result
is therefore narrower than the original design: keep random local rays and the
irrational depth relation; discard the rational-ray and dyadic-scale claims.

Quasi-random projection beats the flat baseline on two corpora and loses on
`README.md`, so it is not a universal improvement. The original rational/dyadic
candidate remains `NO_LIFT` because it loses to the scrambled control on every
corpus.

### Layered Gain Test

Four independently seeded projection arrays were overlaid while retaining 48
features and one 147-parameter head. Three aggregation rules were tested:

- equal-weight feature mean
- fixed-ratio feature sum with weights `1, 1/2, 1/4, 1/8`
- shared-head probability mixture, which combines predictions after projection

The probability mixture is the strongest matched post-projection test:

| Pinned corpus | Single quasi-random | Layered scrambled | Layered quasi-random | Layered minus single |
|---|---:|---:|---:|---:|
| `README.md` | 7.605706 | 7.618338 | 7.618005 | +0.012300 |
| `aethermoore.py` | 6.460641 | 6.473747 | 6.473017 | +0.012376 |
| `docs/GEOMETRIC_REASONING_FRAMEWORK.md` | 7.336792 | 7.354125 | 7.352846 | +0.016054 |

Positive `layered minus single` is worse because lower compression cost is
better. Equal-weight and fixed-ratio feature overlays also lose to the single
view on all three corpora. Layering independent projection maps by averaging
features or probabilities is therefore `NO_LIFT`; it cancels the useful
seed-specific relation signal instead of accumulating it.

Earlier evidence rows with different hashes reported one extra input byte and
are superseded by the raw Git-object rows above. Because moving the single
80-percent split by one byte materially changed some scores, split-position
robustness is required before promoting the remaining directional gain.

### Split-Position Robustness

The same exact blobs were rerun at causal split fractions `0.6`, `0.7`, and
`0.8`. Positive values below are compression improvements in bits/input byte.
`*` means the improvement exceeds `2 * pooled_sd` for that paired run.

| Corpus | Split | Quasi vs flat | Quasi vs scrambled | Random vs rational | Aperiodic vs periodic |
|---|---:|---:|---:|---:|---:|
| `README.md` | 0.6 | +0.012392* | +0.000361 | +0.008266* | +0.003639* |
| `README.md` | 0.7 | +0.017454* | -0.000134 | +0.005066* | +0.001870 |
| `README.md` | 0.8 | -0.008269 | +0.004917 | +0.012009* | +0.005674* |
| `aethermoore.py` | 0.6 | +0.021970* | +0.004342 | +0.010455 | +0.003373* |
| `aethermoore.py` | 0.7 | +0.032953* | +0.004711 | +0.007599 | +0.003036 |
| `aethermoore.py` | 0.8 | +0.024380* | +0.004077 | +0.012490* | +0.001121 |
| `GEOMETRIC_REASONING_FRAMEWORK.md` | 0.6 | +0.013497* | +0.005935 | +0.013028* | +0.002525* |
| `GEOMETRIC_REASONING_FRAMEWORK.md` | 0.7 | +0.002830 | +0.009543 | +0.021750* | +0.003540* |
| `GEOMETRIC_REASONING_FRAMEWORK.md` | 0.8 | +0.038025* | +0.007881 | +0.016521* | +0.003898* |

Quasi-random beats flat in 8/9 positions and scrambled in 8/9, but it clears
the gate against scrambled in 0/9. The full synthesis therefore remains
`UNDERPOWERED`. Random local rays beat rational rays in 9/9 and clear 7/9
gates. Irrational depth beats periodic depth in 9/9 and clears 6/9. These are
component results, not proof that their combination already beats every
equal-budget next-token control.

Layer gains `0, 1, 2, 4`, quadratic gains `0.5, 1, 2`, and multiplicative
self-reinforcement gains `0.5, 1, 2` did not reverse the aligned-versus-scrambled
ordering on the pinned README corpus. These are ablations, not a reason to tune
against the held-out split until it turns green.

A deterministic random-byte guard (`76a118b8dff8`) scores 9.285440 flat,
9.348123 scrambled, and 9.346940 aligned, so the evaluator does not manufacture
a projection win on noise.

A future-token mutation check is part of `selftest`; changing the target and
all later trits leaves the current feature array unchanged.

## Working Rule

Keep three layers separate:

- `encode/decode`: reversible symbolic surface
- `blend/unblend`: reversible multi-tongue channel routing
- `entropy`: measurement of compressibility and surface overhead
- `predict-compress`: cross-entropy comparison against next-token baselines
- `projection-benchmark`: held-out test of the relational projection hypothesis
- `trit-overlay`: ternary class table over binary prefix-code space

Do not optimize the token spellings by vibes. Measure them.
