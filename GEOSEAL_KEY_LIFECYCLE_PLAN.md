# GeoSeal key lifecycle — generation, gating, sealing, and traversal-derived expiry

**Status:** plan. Nothing in this document has been built. Every component it
composes already exists and was executed on 2026-08-20; the citations below record
what was actually run, not what is claimed.
**Scope:** replace the GeoSeal key fallback, and give key material a lifecycle whose
expiry is derived from layer traversals rather than from a wall clock.
**Non-goal:** new cryptography. This composes existing primitives and adds one new
mechanism (the traversal ratchet). No primitive is invented, weakened, or replaced.

Kept as a local file rather than published anywhere: it names a live, currently
exploitable defect in a running system. That belongs on disk, not on a host.

---

## 1. The problem, stated exactly

`six-tongues-geoseal/aethermoore.py:1567`

```python
def _geoseal_key() -> bytes:
    key = os.environ.get(GEOSEAL_SECRET_ENV)     # "GEOSEAL_KEY"
    if key:
        return hashlib.sha256(key.encode()).digest()
    # Deterministic default for reproducible selftest
    return hashlib.sha256(b"scbe-aethermoore-default-key-do-not-use-in-production").digest()
```

`GEOSEAL_KEY` is unset on this machine. Therefore every seal is HMAC'd under a key
**published in the source tree**. Measured consequence:

- `~/.clay/training-studio/sessions/default/state.geoseal.json` is
  `{context, mac, payload_hex}` with `mac = HMAC-SHA256(key, aad || payload)`.
- Anyone holding the repository can recompute that MAC over any payload.
- The seal therefore proves *no accidental corruption* and **nothing about authorship**.

This is not a weak key. It is a public one. The literal says so itself.

**The defect is the silence, more than the key.** A missing key does not fail — it
seals anyway and prints a healthy status line. Two sibling instances of the same
pattern exist and are in scope:

| # | Location | Silent fallback | Consequence |
|---|---|---|---|
| 1 | `aethermoore.py:1567` `_geoseal_key` | published literal | forgeable MAC |
| 2 | `pqc/pqc_core.py:161` `_MockKyber` / `_MockDilithium3` | hashlib mock | a hash is not a KEM; no unforgeability |
| 3 | `scbe_14layer_reference.py:43` `layer_0_intent_modulation(t, key="default_key")` | literal default | intent modulation unbound to any identity |

liboqs 0.15.0 **is** installed and real here (Kyber768, ML-KEM-512, ML-DSA-44/65/87
all enabled), so #2 is currently dormant — but it is dormant by luck, not by gate.

---

## 2. Inventory — what exists, with its evidence

Classifications and test counts are quoted from
`patent-evidence-20260820/10_current_evidence_20260820/08_storage_design_lineage.md`,
which is an audit, not a brochure. Everything marked RUN was executed 2026-08-20.

| Component | Path | Evidence | Role here |
|---|---|---|---|
| DPAPI protect/unprotect | `SCBE/src/security/secret_store.py` | **RUN** — round-trip true; emits `AQAAANCMnd8…` blob | root of trust |
| Braid vault | `SCBE/src/crypto/braid_vault.py` | 21 primitive tests passed; **RUN** — deterministic, 32-byte, inverses valid | structure + domain separation |
| Sacred Eggs | `SCBE/src/crypto/sacred_eggs.py` | 44 tests passed | container: yolk / shell / albumen |
| Sacred Egg registry | `SCBE/src/symphonic_cipher/scbe_aethermoore/sacred_egg_registry.py` | 16 tests passed | lifecycle states + ritual log |
| Sacred Egg integrator | same commit `d337027` | 42 tests passed | GeoSeal binding, ring descent |
| RWP-v3 | `SCBE/src/crypto/rwp_v3.py` | 7 tests passed (sealed memory packets) | authenticated encryption w/ AAD |
| Real PQC | liboqs 0.15.0 | **RUN** — ML-DSA-65 enabled | optional cross-boundary signature |
| 14-layer reference | `SCBE/src/scbe_14layer_reference.py` | L0–L14 implemented | the traversal |
| Decimal drift | `loom/clay_drift.py` | **RUN** — SELFTEST PASS | traversal witness |

### 2.1 Constraints inherited from the audit — these bound the design

The storage audit is explicit about what its own components do *not* prove. Those
lines are load-bearing and are carried into this design rather than argued with:

- Braid vault: *"functional prototype; **do not rely on an unproved braid-hardness
  claim**."* → **The braid is not the security boundary.** It supplies structure and
  domain separation. Strength comes from HKDF-SHA256, HMAC, and DPAPI.
- Sealed memory packets: *"**tokenizer is not the security boundary**."* → the six
  tongues route and label. They do not protect.
- Tri-lattice membrane: *"**fallback is not a security proof**."*
- *"It does not support a claim that every geometric visualization is
  cryptographically protective."*

If any step below appears to rest on braid hardness or on tongue structure for its
security, that step is wrong and should be rejected in review.

---

## 3. Architecture

Mapped onto the element numbering the patent audit already uses (810–870), so this
extends the existing tree rather than starting a parallel one.

```text
DPAPI keystore  (OS-bound, no plaintext, not a file in the repo)
  └── master seed S          32 bytes, CSPRNG, never leaves DPAPI in the clear
  └── braid word W           the "combination lock" — a topological path, not a string

           │  (both required)
           ▼
  braid derivation  850     _init_strands(S) → _apply_braid(W) → _finalize()
      3 strands over tongue PAIRS:  KO×AV presence · RU×CA structure · UM×DR depth
      dual primitive per strand:    SHA3-256  +  BLAKE2b-256          ← "multi hash"
           │
           ▼
  Sacred Egg  850           yolk = braid output
      shell   = SHA-256(yolk ‖ context)     ← PUBLIC. the only value written in clear
      albumen = HKDF(yolk, salt, info)      ← per-purpose, derived on demand, never stored
           │
           ▼
  incubation  850           solitary_incubation(cycles=3): iterated HKDF,
                            each cycle folds in the previous  ← second "multi hash"
           │
           ▼
  K_geoseal                 albumen label "geoseal", ratcheted by traversal (§5)
                            in memory only, for the life of one call
           │
           ▼
  lifecycle + receipts 860  registry status · traversal counter · drift receipt chain
```

**What is on disk, in the clear, and safe to be:** the shell (a commitment, not a
secret), the traversal counter, the drift receipt chain, the ritual log.
**What is on disk encrypted:** nothing that is not also in DPAPI.
**What is never on disk:** S, W, the yolk, any albumen key, `K_geoseal`.
**What is never in a context file:** all of the above. No `.env`, no `CLAUDE.md`, no
`AGENTS.md`, no `MEMORY.md`, no committed config. An AI reading every file in this
workspace must find no key material. That is a hard requirement, and §8 tests it.

---

## 4. The fourteen gates

The traversal is `scbe_14layer_reference` L0 → L14. Each layer carries a check that
can refuse. A refusal is **fail-to-noise** (patent innovation 3): the caller receives
entropy, never a description of what failed.

| Layer | Function | Gate |
|---|---|---|
| L0 | intent modulation | reject the literal `"default_key"`; intent must bind to the egg shell |
| L1 | complex state | dimension `D` matches the sealed manifest |
| L2 | realification | no NaN/Inf entering the real embedding |
| L3 | weighted transform | `G` is the pinned matrix (hash-checked), not a caller's |
| L4 | Poincaré embedding | ‖u‖ < 1 − eps_ball; refuse a point on or past the ball boundary |
| **L5** | **hyperbolic distance** | `d_H` finite; **drift probe A** (acosh — hard transition) |
| L6 | breathing transform | `b ∈ [b_min, b_max]`; the threshold breathes but stays bounded |
| **L7** | **phase transform** | phase deviation within tolerance; **drift probe B** (exp) |
| L8 | realm distance | a realm is actually selected; no silent default realm |
| L9 | spectral coherence | coherence ≥ floor, else the signal is not a signal |
| L10 | spin coherence | phasor set non-degenerate |
| L11 | triadic temporal | the three time slices are ordered and distinct |
| **L12** | **harmonic scaling** | wall `φ^d_H` finite; **drift probe C** (pow) |
| L13 | risk decision | ALLOW / QUARANTINE / DENY at θ₁=0.33, θ₂=0.67. Only ALLOW advances |
| L14 | audio axis | axis present or explicitly declared absent — never inferred |

L5, L7 and L12 are the three probes because `clay_drift.py` measured drift to be
largest exactly there ("exp, acosh, pow: L5 d_H, L7 phase, L12 wall"). They are the
layers that do the hardest arithmetic, so they carry the most signal per byte.

A DENY or QUARANTINE at L13 does **not** advance the ratchet. Only a full clean
traversal counts. That makes the counter a record of successful authorized work.

---

## 5. Traversal-derived expiry — the one new mechanism

### 5.1 Four clocks, not one

A single clock is always evadable, because an attacker only has to beat one rule.
The system therefore runs **four clocks with different advance conditions**, and all
four already exist:

| Clock | Advances on | Where | Evades by |
|---|---|---|---|
| **wall** | absolute elapsed time | `sacred_egg_registry.ttl_seconds` | waiting — but waiting expires you |
| **layer** | one *clean* L0→L14 traversal, ALLOW at L13 | new (§5.2), counter in registry | doing legitimate authorized work |
| **attempt** | every try, success **or** failure | `registry.log_attempt` / `get_attempts`; `agentic/quarantine_lock.py` tiers ALLOW/REVIEW/QUARANTINE/ESCALATE/DENY at budgets 512/128/16 | not retrying |
| **dilated** | per-identity query rate | `cpse.py` `lorentz_factor` / `VirtualGravityThrottler` | going slow |

### 5.1.1 The time-dilation trap

`cpse.py` implements virtual gravity as a real Lorentz factor, with query rate in the
role of velocity and `rho_critical` in the role of *c*:

```
    γ(ρ_E) = 1 / √(1 − (ρ_E / ρ_critical)²)          Δt = t_base × γ
    ρ_E → ρ_critical  ⇒  γ → ∞                       (event horizon)
```

This is the same asymptotic form as the patent's harmonic wall
(`1/(1 − d/horizon) → ∞` at the boundary) — one stated over distance, this one over
rate. Structural consistency, not coincidence.

**The trap is the interaction between the clocks, not any one of them.** Every
strategy against one clock is a losing strategy against another:

- **Probe fast** → ρ_E rises → γ rises → each attempt costs more *wall* time.
  At ρ_E ≥ ρ_critical, γ = ∞ and no answer is ever returned. The attacker's proper
  time dilates while the system's does not.
- **Probe slow** to stay under ρ_critical → the wall clock runs out first.
- **Probe many times** → the attempt clock trips quarantine, and budgets collapse
  512 → 128 → 16 tokens.
- **Probe cleverly** → the layer clock still does not move, because it advances *only*
  on a clean traversal with ALLOW at L13.

The decisive asymmetry: **failed attempts advance three clocks and never the fourth.**
A prober burns wall time, attempt budget, and dilation penalty while the layer clock
stands still — so the key expires out from under them mid-attack, and re-keying mints
a new shell they have never seen. A legitimate caller at low rate sees γ ≈ 1, no
delay, and advances the layer clock on every clean pass.

Wall-clock TTL alone would have the opposite property: it expires a key that was never
used, and fails to expire one used a million times. Exposure accumulates with **use**,
which is what the layer clock measures — and with **pressure**, which is what the
attempt and dilated clocks measure.

### 5.2 The ratchet

One clean traversal produces a drift vector
δₙ = (|Decimal − float64| at L5, L7, L12), computed the way `clay_drift.py` computes
it: the same pipeline twice, once in 50-digit `Decimal` (reference) and once in
`float64` (working).

```
    K₀      = albumen("geoseal")                       from §3
    Kₙ₊₁    = HKDF-SHA256(ikm = Kₙ,
                          salt = H(quantize(δₙ)),
                          info = b"geoseal:traversal:" ‖ n+1)
    receipt = (n, H(quantize(δₙ)), shell, status)      appended to the chain
```

Forward-only: HKDF is one-way, so `Kₙ` cannot be recovered from `Kₙ₊₁`. A compromised
key at traversal *n* does not expose traversals before *n*.

### 5.3 Why the counter cannot be forged

A bare counter can be rewound by editing a number. This one cannot, because
**advancing it requires the drift vector, and the drift vector requires actually
running the layers.**

- Cannot skip ahead: δₙ is unknown without executing traversal *n*.
- Cannot rewind: `H(quantize(δₙ))` is committed to the receipt chain.
- Cannot fake a traversal: a δ inconsistent with the pipeline yields a different
  `Kₙ₊₁`, and the next seal fails to verify. It fails *loud*, at the seal.

This is the property `clay_drift.py` already demonstrates, in its own numbers:

```
endpoint gap (what an ANSWER check sees):     0.00e+00   → passes both
drift-trajectory gap (what PROCESS sees):     4.47e+00   → different process
```

Same endpoint, different route, caught. The drift binds the key to the **path**, not
to the result.

### 5.4 Expiry

`K_geoseal` is valid for **N traversals** (proposed default N = 64, one per opcode,
tunable per context). At *n = N* derivation refuses and the registry moves the egg
`SEALED → EXPIRED`. Re-keying mints a new egg with a new shell and a lineage pointer
to its parent — `SacredEgg.create(parent_ids=[…])` and
`EggCarton.get_lineage_chain()` already support this.

**Four independent expiry conditions, whichever fires first** — one per clock (§5.1):

1. **layer** — traversal budget exhausted (`n ≥ N`). The new mechanism.
2. **wall** — `ttl_seconds` exceeded. Existing registry TTL, retained.
3. **attempt** — failure count crosses the quarantine tier. Existing `log_attempt`
   plus `quarantine_lock`; an egg that reaches DENY does not get to keep its key.
4. **dilated** — sustained ρ_E ≥ ρ_critical for an identity. Existing
   `VirtualGravityThrottler`; the event horizon is itself an expiry event, not just
   a delay, because a caller who can never be answered has no live key.

Only condition 1 is new. The other three are existing mechanisms given a shared
lifecycle vocabulary, so that a single `status()` call answers "is this key alive?"
against all four rather than against whichever one a caller happened to check.

### 5.5 The engineering hazard, stated up front

**IEEE-754 transcendentals are not portable.** `exp`, `acosh` and `pow` may differ in
the last bits across libm versions, CPU architectures, and compiler flags. Raw δ is
therefore *not* a stable cross-machine value, and hashing it raw would make a key that
works on one machine and silently dies on another.

Mitigations, in order of preference:

1. `quantize(δ)` — round to a coarse band (proposed: 6 significant decimal digits)
   before hashing, so the witness survives last-bit variation while still separating
   genuinely different processes. The measured gap is 4.47 against 0.00; six digits is
   many orders of margin.
2. Bind the receipt to a platform fingerprint, so a cross-machine mismatch reports
   "different platform" rather than "tampered".
3. Pin the reference side to `Decimal` (already deterministic at `prec=50`) and treat
   only the float64 side as platform-variant.

**This must be measured before the design is trusted**, and it is the first
experiment in §7. If quantized δ does not reproduce across this laptop and
`sidekick`, the ratchet is single-machine and the document must say so.

---

## 6. Migration — ordered, and the trap

> **Do not set `GEOSEAL_KEY` and restart.** `_load_state` → `geoseal_decrypt` raises
> `ValueError("GeoSeal MAC mismatch")` with no fallback. The existing state was sealed
> under the demo key. Changing the key first makes 136 KB of studio state and 21
> events unreadable.

| Step | Action | Gate before proceeding |
|---|---|---|
| 0 | Copy `state.geoseal.json` and `events.jsonl` to a dated backup outside the session dir | backup hashes recorded |
| 1 | `geoseal_keygen.py init` — CSPRNG seed + braid word → DPAPI; write shell only | shell present, no secret on disk (§8 scan) |
| 2 | Dry-run re-seal: decrypt under demo key, re-encrypt under derived key, to a **temp path** | temp file decrypts under the new key AND payload is byte-identical to the original |
| 3 | Atomic swap of `state.geoseal.json` | studio `--status` loads and reads `owner key` |
| 4 | Replace the `_geoseal_key` literal with a loud refusal | selftest that previously passed on the literal now refuses |
| 5 | Gate `pqc_core` on `is_liboqs_available()`; mock only under an explicit test flag | importing with liboqs hidden raises rather than mocks |
| 6 | Bind L0 intent to the egg shell; remove `"default_key"` | L0 with no key refuses |
| 7 | Enable the traversal ratchet at N; leave `ttl_seconds` as backstop | receipt chain grows by one per clean traversal |

Step 2 is reversible; step 3 is the first irreversible one, which is why the
byte-identity check sits before it and not after.

---

## 7. Separately: the event-chain fork

Not caused by the key, and fixed differently. Diagnosed 2026-08-20:

```
6  word_meaning_training_started
├─ 7  → 9 → 10 → … → 20     (the line that continued)
└─ 8  word_meaning_training_imported   (orphan; nothing builds on it)
```

All 21 events have **valid self-hashes** — nothing was tampered with. Events 7 and 8
both claim event 6 as parent: two concurrent writers, one head, no lock. The data is
a DAG; `verify_events` walks the file line-by-line assuming a list.

**Fix the verifier, never the ledger.** Resolve `previous` by hash lookup, require
exactly one head, and *report* a fork explicitly. Re-linking events 8+ would require
re-hashing them, which destroys the tamper-evidence the ledger exists to provide.

Add the write lock that `Context-Ledger/context_cloud.py` already got for the
identical failure: `LK_NBLCK` in a retry loop with backoff.

*(Disclosure: event 20, `conversation`, 19:42 on 2026-08-20, is mine — a
`/representations` call that MSYS path-translation turned into conversational input.
Benign, on the main line, hashes cleanly. It is my write in this history.)*

---

## 8. Test plan — what must pass before this is believed

1. **No-secrets scan.** Grep the entire workspace for the shell, and for any 32-byte
   high-entropy literal. The shell may appear; nothing else may. Run it against
   `CLAUDE.md`, `AGENTS.md`, `MEMORY.md`, every `.env`, and the git index.
2. **Refusal tests.** Each of the three silent fallbacks must now raise. Assert on the
   exception, not on a log line.
3. **Cross-machine drift reproduction** (§5.5) — this laptop vs `sidekick`, same input,
   compare `quantize(δ)`. **This gates the whole ratchet.** If it fails, the mechanism
   is single-machine and must be documented as such rather than quietly shipped.
4. **Ratchet monotonicity.** `Kₙ₊₁ ≠ Kₙ`; `Kₙ` is not recoverable from `Kₙ₊₁`;
   replaying an old δ does not reproduce a later key.
5. **Expiry.** At `n = N` derivation refuses and the registry reads `EXPIRED`.
6. **Gate refusals.** Each of the 14 gates, forced to fail, returns noise — and the
   noise is not distinguishable per-layer (that is the point of fail-to-noise).
7. **Round-trip under the new key**, and **failure under the old**, on a copy.
8. **Existing suites still green:** sacred eggs 44, registry 16, integrator 42,
   braid-vault primitives 21, RWP-v3 7. Run separately — the audit records that a
   combined six-file run exceeded 120 s and braid-vault alone exceeded 180 s, so a
   slow KDF suite must not be allowed to hide unrelated results.

---

## 9. What this does not claim

- **Decimal drift is integrity-of-process, not authentication.** An adversary holding
  the code can also run the fourteen layers and produce a valid trajectory. It proves
  *this path was taken*; it does not prove *who took it*. It complements the keyed
  stack and does not replace it.
- **The braid is not the security boundary** (§2.1). Remove the braid and the scheme
  is still HKDF/HMAC/DPAPI. Remove HKDF and it is nothing.
- **DPAPI binds to this Windows user on this machine.** That is the right root for a
  local studio seal. It is not portable, and it is not a substitute for asymmetric
  signing if a seal must ever be verified off-box — for that, use the real ML-DSA-65
  that liboqs already provides, and say "signed" only then.
- **An HMAC seal verified by the holder of the key proves only that the holder sealed
  it.** `toolkit.py` already records this limit: *"a holder of the live Toolkit could
  recompute, so for cross-boundary sign externally."* The same caveat applies here and
  is not weakened by any amount of braid, tongue, or geometry.
- **No claim is made that geometric structure is cryptographically protective.**

---

## 9a. The canary — turning the defect into a trap

**Issac, 2026-08-20: "turn the defect into a trap."**

Deleting the published literal wastes it. An attacker who already read it simply
learns nothing and moves to the next thing. Kept as bait, it becomes the one asset a
defender rarely has: **an identifier that only an attacker would use.**

### Two-key regime

| Key | Derived from | Protects | Detects |
|---|---|---|---|
| `K_real` | DPAPI → braid → egg albumen (§3) | everything real | — |
| `K_canary` | the published literal, unchanged | **nothing, ever** | anyone who read the source and used it |

Verification order on any inbound seal:

1. `K_real` verifies → normal path. Advance the layer clock.
2. `K_canary` verifies → **trap trip** (below).
3. Neither → ordinary MAC refusal.

### Discriminating a misconfigured friend from a forger

A canary MAC alone cannot tell "our own script forgot to set the key" from "someone
is forging". The drift witness (§5) separates them, and this is where the two stacks
earn their keep together:

| Canary MAC | Drift trajectory | Verdict | Response |
|---|---|---|---|
| valid | valid, consistent with the receipt chain | **misconfiguration** — they really ran the fourteen layers | warn loudly, do not punish, tell them to re-key |
| valid | absent, or inconsistent | **forgery** — they had the key but never did the work | trap |

A forger can compute the MAC, because the key is public. They cannot cheaply produce
a drift trajectory consistent with the chain without actually traversing the layers.
That asymmetry is the whole discriminator.

### Response to a trip — silent

- **Record**: identity, timestamp, payload digest, context tag, the tongue route, and
  what they attempted to open or seal. Into `registry.log_attempt`, whose failure
  record is already *"deliberately generic"* on the caller-facing side.
- **Return**: fail-to-noise (patent innovation 3) shaped to look like ordinary
  success. **Do not announce the trip.** A trap that says "caught you" teaches the
  attacker exactly where the boundary is; a trap that returns plausible garbage costs
  them time and tells them nothing.
- **Advance the attempt clock**, escalate the quarantine tier (512 → 128 → 16).
- **Feed ρ_E**, so the dilated clock pulls them toward the event horizon.
- **Never advance the layer clock.**

That last line is the trap closing. Per §5.1.1, a trip advances three clocks and never
the fourth — so the forger burns wall time, attempt budget, and dilation penalty while
the layer clock stands still, and `K_real` expires out from under them mid-attack.
Re-keying then mints a new shell they have never seen.

### Same shape for the other two defects

The pattern generalises to both siblings in §1:

- **`pqc_core` mock**: a caller who successfully completes a KEM against the hashlib
  mock has either misconfigured liboqs or is probing. Same table, same response.
- **`layer_0_intent_modulation(key="default_key")`**: any intent arriving modulated
  under the literal default is bait-tripped at L0 — the first gate, before any work.

### Hard bounds — the trap does not replace the fix

1. **`K_canary` must never protect anything real.** Any state that verifies only under
   the canary is quarantined and never loaded as authoritative. The studio must refuse
   to boot from canary-sealed state.
2. **This is detection, not prevention.** It is strictly additive to §6 steps 1–4. A
   honeypot that is also your production key is just a vulnerability with logging
   attached. `K_real` must be live before `K_canary` is armed, never after.
3. **The bait's value is that it is public.** Do not rotate it, do not obscure it, do
   not remove it from history. It is already distributed; that is the asset.
4. **Log the trip, never the payload's secrets.** Digests and routes only — the same
   rule the ledger already runs under.

## 10. Build order

1. `geoseal_keygen.py` — init / derive / status. No key ever printed or returned to a
   log. (§3)
2. The drift-reproduction experiment. (§8.3) — **before** the ratchet is wired, because
   a negative result changes the design.
3. The three loud refusals. (§6 steps 4–6)
4. Re-seal migration, backed up and reversible. (§6 steps 0–3)
5. DAG-aware `verify_events` + write lock. (§7)
6. Traversal ratchet and registry integration. (§5)

Steps 1–4 are independent of the ratchet and fix a live exploitable defect. They
should land first even if §5 is deferred.
