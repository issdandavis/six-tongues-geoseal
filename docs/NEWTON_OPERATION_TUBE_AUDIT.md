# Newton operation tube: exact claim and model gate

The operation tube is an exact coordinate change over a fixed-length integer
sequence.  For samples `x[0] ... x[n-1]`, it stores

```text
c[k] = Δ^k x[0]
x[i] = Σ(k=0..i) binomial(i, k) c[k]
```

The transform is triangular with ones on its diagonal.  Keeping `n` and every
coefficient therefore gives a bijection over integer sequences of length `n`.
The six tongue strings remain six permutations of the same byte board; they do
not add six independent measurements.  Canonical balanced ternary is another
exact spelling of each signed coefficient.

## Independent audit, 19 October 2026

The implementation passed 1,000 seeded random round trips and 63,382 prefix
checks.  A coefficient at order `k` depends only on bytes `0..k`, so appending a
future byte does not change any earlier coefficient.

The full coefficient vector is not a general compressor.  Ideal feature size
below counts each balanced trit as `log2(3)` bits and excludes JSON framing.

| board | bytes | degree | largest coefficient | ideal size / input |
|---|---:|---:|---:|---:|
| random | 16 | 15 | 20 bits | 1.75x |
| random | 64 | 63 | 68 bits | 4.62x |
| random | 128 | 127 | 132 bits | 8.72x |
| random | 256 | 255 | 260 bits | 16.64x |
| random | 512 | 511 | 514 bits | 32.57x |
| this repository's README prefix | 512 | 511 | 514 bits | 32.37x |
| repeating `abc ` | 512 | 511 | 516 bits | 32.67x |
| byte ramp `0..255` | 256 | 1 | 1 bit | 0.20x |

Low-degree boards are the useful special case.  Ordinary text and irregular
bytes normally use degree `n-1`; their high-order differences grow rather than
compress.  The tube is therefore an exact audit view and a source of candidate
features, not evidence of faster token processing or learned capability.

The public `operation_tube` surface is capped at 512 bytes.  Longer payloads
should be windowed with explicit offsets and a hash of the whole board.  The
leading-coefficient implementation keeps one difference row at a time, reducing
working memory from quadratic to linear, while exact runtime remains quadratic.

## Model-facing candidate

Use bounded local differences instead of feeding all high-order coefficients:

```text
d[k, i] = Δ^k x[i-k], for k <= i
```

At token position `i`, these features only read positions through `i`.  For byte
inputs, order zero is bounded by 255 and order `k >= 1` is bounded by
`255 * 2^(k-1)`; orders zero through two stay bounded at 255, 255, and 510.
Missing history is represented by `None`, not zero.  A first
difference placed at position `i-1` would reveal token `i`, so training code must
use the alignment returned by `causal_local_differences`.

The current Clay architecture and Python phases are preregistered and must not be
changed mid-run.  A later experiment can compare, with identical parent weights
and parameter counts:

1. byte baseline;
2. aligned order-0/1/2 local differences with input access;
3. the same features shuffled between rows;
4. a size-matched learned projection receiving no difference values.

Use at least three paired seeds.  The aligned arm must beat the parent, byte
baseline, shuffled control, and size-matched control by more than twice pooled
seed standard deviation with every paired sign positive.  Report held-out code
execution, composition, byte loss, and story retention separately.  Falling
training loss or exact reconstruction of the tube is not a capability result.

## Reproduce

```powershell
python -m unittest test_newton_tube.py -v
python -m unittest discover -v
python aethermoore.py selftest
python aethermoore.py streams --hex 0103060a0f
python audit_newton_tube.py
```
