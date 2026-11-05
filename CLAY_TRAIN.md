# Clay Training Studio

`clay.train` opens a local conversational manager around Clay's existing real
trainers. It does not replace those trainers, generate shell commands, or claim
that a reversible codec is itself a neural network.

Install the terminal command once:

```powershell
cd C:\dev\six-tongues-geoseal
.\install_clay_train.ps1
clay.train
```

The same surface is part of GeoSeal:

```powershell
python aethermoore.py train
python aethermoore.py train --status
```

Set `GEOSEAL_KEY` in the local environment to bind receipts to an owner key.
Without it, GeoSeal uses its documented reproducible demo key. GeoSeal supplies
integrity and context binding here, not encryption; approved dialogue can be
exported to ordinary JSONL with `/export` when a trainer needs it.

## A real session

```text
clay.train> I want the first layers to curve, interweave, twist, and run backwards
[routed] {"geometry":["curve","interweave","twist","reverse"]}
clay.train> /accept
clay.train> /inject 1
clay.train> /voice train 5
clay.train> /voice adopt latest
clay.train> /voice say What did you learn?
clay.train> /hatch 3
clay.train> /plan
clay.train> /run swarm
```

Plain conversation updates the local goal and maps architecture words to
existing surfaces. `/accept` is the deliberate learning boundary: it writes one
canonical prompt/response record in a forward KO representation and a reversed
DR representation, then proves that both recover the same bytes. The two views
share one SHA-256 joint.

`/run` invokes
`C:\dev\clay-quines\home\local_train\local_train.py` as an argv list. The
existing lane owns training, held-out grading, checkpoints, and its append-only
journal. Clay Training Studio never passes `--promote`.

`/voice train` invokes Clay's own
`C:\dev\clay-quines\home\ingest\dialogue_quine\dialogue_quine.py`. It is a
small B256 language model, not an Ollama or Hugging Face wrapper. Its first
layers follow two nonlinear paths (`tanh` and `sin`), flip and interleave their
features, then feed a recurrent core. Each accepted exchange is trained in a
forward and byte-reversed orientation.

Generation zero trains the whole dialogue model. A later run loads the active
checkpoint, freezes its inherited tensors, and adds one zero-gated growth ring
with access both before the recurrence and before the output head. At zero the
ring reproduces the parent logits exactly; during training only the new ring
can move. This is the executable `a -> A -> offspring(A) -> B` lifecycle: the
core is conserved, lifetime learning becomes a parent-linked delta, and the
child begins from the mature parent rather than from `a` again. Checkpoints
carry parent, dataset, genome, and model-state hashes.

Each growth ring also contains a learned token-local relay. Training compares
that routed ring to an untrained baseline, a shuffled-response control, and an
equal-parameter always-on ring. A candidate becomes the active parent only if
it clears all three comparisons by more than twice pooled seed SD. The owner is
not locked out by that evidence gate: `/voice adopt latest` explicitly selects
any hash-verified candidate, while preserving the evidence verdict in its
promotion receipt. `/voice say-candidate` inspects a candidate without adopting
it.

## J-space conveyor contract

Clay's existing J-space notes define a shared representational space; the 2026
Jacobian-lens paper uses the same name more narrowly for sparse verbalizable
directions inside a language model. This CLI does not silently equate them. It
implements a testable bridge: every approved semantic package has a plain-word
form plus typed KO and reversed-DR routes. The tokenizer is the conveyor, the
digits/opcodes are its gears, and the meaning is the package. Both routes must
decode to the same canonical bytes before the example becomes nutrient.

`/learn PATH` bulk-loads owner-approved JSONL or JSON-array rows containing
`prompt` and `response`. It redacts, deduplicates, compresses, seals, and proves
both semantic routes for each row. This supplies more packages without storing
duplicate belt representations.

The neural map is **not** described as bijective. Bijectivity belongs to the
paired data codec. The neural evidence comes separately from a routed-real arm,
an untrained baseline, a shuffled-response control, and a parameter-matched
always-on-ring control on a prompt-disjoint holdout. With fewer than four
approved pairs the result is explicitly `TRAIN_ONLY_SMOKE`.

## Opcode micro-hatchery

`/opcode hatch smoke` runs the complete mechanism cheaply. `/opcode hatch
local` runs the current laptop regime: a fixed population of four six-byte
genomes, binary tournament selection, two-point crossover, elitism, and a
Hamming-homogeneity restart. Each genome controls neural width/depth and sparse
per-layer pair/hyperthread routes around the fixed 64-opcode substrate. The
candidate model itself still learns by ordinary PyTorch backpropagation.

Search opcodes and promotion opcodes are disjoint. Promotion compares the
evolved route over multiple seeds with a same-density randomized route, the
fixed charged graph, the flat graph, and the untrained state. It writes the
genome, route tensors, traces, controls, report, and cloud handoff together.
The handoff is inert: no Kaggle or Lightning job is launched by this command.

```text
clay.train> /opcode hatch smoke
clay.train> /opcode hatch local
clay.train> /opcode status
```

## Hash-verified cluster parent

`/opcode cluster import POINTER.json` attaches a completed `clay-cluster`
checkpoint through the existing GeoSeal CLI. The command verifies the pointer,
checkpoint, training data, complete 64-by-4 direct-opcode grid, broad held-out
evaluation, and selection decision by SHA-256 before writing a sealed receipt.
It refuses to write when the session event journal is broken.

The imported checkpoint remains a scoped parent. Selection means that it may
seed the next direct opcode-control experiment; it does not grant tool or
compiler authority, claim conversational ability, or turn a missed broad gate
into a promotion. Use a dedicated single-writer session for each lineage:

```powershell
clay.train --session cluster-binding-20261104 --command "/opcode cluster import C:\dev\clay-cluster\STABLE_OPCODE_PARENT.json"
clay.train --session cluster-binding-20261104 --command "/opcode cluster status"
clay.train --session cluster-binding-20261104 --status
```

## Measured representation seams

`/representations` exposes the 2026-08-05 census across `C:\dev` and
`SCBE-AETHERMOORE`. The four farm translation lanes are `clay-one-repo`,
`clay-quines`, `clay_home`, and `clay-wafer-net`; they already co-locate five or
six of tongues, trits, cube, opcode, bytes, and HF-BPE. Training-transfer tests
belong at those existing seams.

STISA is different: its implementation remains owned by
`C:\Users\issda\SCBE-AETHERMOORE\python\scbe\atomic_tokenization.py`. The plan
records its live path and hash as an external anchor and does not duplicate it
into Clay. The census measures references, not runtime dependencies, so it is a
routing map rather than evidence that a model has learned every representation.

## Kaggriculture reasoning exam

Kaggriculture is downstream evaluation, not training nutrient. The registered
target is submission `55283743` (`main_profit.py`). Its public rating is live
and has changed without a new submission, so no score is hard-coded. The strict
external rule compares a candidate and target only when both appear with
complete numeric scores in the same authenticated Kaggle CLI response.

The local command runs the existing game environment on fresh seeds, in both
seats, against the exact local `main_profit.py` source and requires the direct
win signal to exceed twice its standard error:

```text
clay.train> /benchmark kaggriculture status
clay.train> /benchmark kaggriculture refresh
clay.train> /benchmark kaggriculture refresh CANDIDATE_SUBMISSION_ID
clay.train> /benchmark kaggriculture gold main_goldroute_flat.py
clay.train> /benchmark kaggriculture local main_feedback_quine.py 12 22000
```

`refresh` is read-only and stores a GeoSeal-bound same-snapshot receipt. It does
not submit. `gold` profiles the three owner-selected winning replay trajectories
and attaches hash-compatible held-out local receipts; it also does not train or
submit. Requested actions and observed state transitions remain separate in the
profile so execution noise cannot be mistaken for intent.

That local result is deliberately labelled a surrogate. Even a local pass is
`LOCAL_PASS_EXTERNAL_UNTESTED`; it cannot be promoted into a claim about the
Kaggle score. The benchmark command neither trains on Kaggriculture nor submits
anything.

## Hatchery lifecycle

The StarCraft terms are a resource scheduler, not claims about biology:

| Role | Clay meaning |
|---|---|
| Hatchery | Persistent session, genome receipt, and candidate queue |
| Larva | Uncommitted capacity for a candidate or worker |
| Hatchling | A bounded one-dimensional training candidate |
| Drone | A successful run/artifact that can do work or seed a workspace |

One Hatchery starts with three Larva and one reserve Drone. A Drone can morph
into a Hatchery only after another reserve Drone exists. That keeps the exact
owner invariant: at least one Hatchery and one spare Drone remain, subject to
the compute and storage values the owner sets with `/budget`.

Candidates are emitted as `0,+1,-1,+2,-2...` offsets from the current axis
centre. A generation moves the centre only when the best direction beats the
centre by more than twice the pooled seed standard deviation and both have at
least three seeds. Anything smaller is recorded as `UNDERPOWERED`, not as a
failure or success.

## Existing geometry bindings

| Word | Existing implementation | Status |
|---|---|---|
| curve | `cross_plane_relations` | trainable |
| interweave / twist | `dynamic_braid` + `ribbon_twist.py` | trainable + diagnostic |
| invert / reverse | `inverse_routing` | trainable |
| dual | `asym_mirror_routing` + `dual_channel.py` | trainable lane + diagnostic |
| palindrome | paired KO/DR record in `clay_train_cli.py` | executable codec |
| copy / replay | quine lifecycle + local training journal | executable |
| wafer | phase-conjugate wafer tests | executable diagnostic |
| stretch | neural strip/wafer layer map | design-only |

The plan artifact carries those labels so a diagram cannot silently borrow the
credibility of a trainer. `C:\dev\clay-wafer-net\train_lightning.py` remains a
plan-only launcher and is not invoked here.

## Commands

Run `/help` for the complete list. The core loop is:

1. talk and set the goal;
2. `/accept`, `/teach`, or `/learn PATH` owner-approved examples;
3. `/voice train 5` to grow a measured dialogue generation;
4. inspect with `/voice say-candidate`, then accept evidence promotion or use
   `/voice adopt latest` as an explicit owner decision;
5. `/hatch 3` paired hill candidates for the other neural lanes;
6. `/plan` for exact commands and sealed receipts;
7. `/run swarm` for real multi-seed held-out measurements;
8. `/opcode hatch smoke|local` for bounded opcode-route evolution;
9. `/benchmark kaggriculture gold ...` and `local ...` for separate downstream
   behavioral and held-out reasoning evidence, never as a training objective;
10. inspect `/status`, then repeat.

A local Ollama model may join as a peer with `/peer use MODEL`; it is optional.
The manager remains usable without Ollama, Unsloth, Hugging Face, cloud APIs, or
any downloaded base model.
