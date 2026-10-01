# ARX gateway and critic: known issues

Status: review record, 2026-09-23. These are documented issues and limitations,
not authorization for immediate code changes. The current gateway toolchain and
critic interruption smoke tests remain useful evidence of their tested paths.
The proposed [deployment runner](arx-deployment-runner-plan.md) is future work.

## KI-01: decision budgets do not bound rejected attempts

`robots/arx/gateway/session_core.py:execute()` returns validation rejections
before incrementing `decisions` or consuming recovery decisions. Exhausting the
accepted-operation decision budget rejects subsequent requests without ending
the episode. `_error()` labels these validation failures `correct_request`.
`robots/arx/gateway/episode.py:EpisodeDriver.run()` can continue indefinitely on
rejections, including automatic nominal Zeva submissions.

Reproduced with `max_decisions=1`: three invalid requests left the count at zero;
one valid request exhausted the budget; the next was rejected while the session
remained `RUNNING_NOMINAL`. Impact: unbounded agent calls or request loops.

Future resolution: count bounded decision attempts separately from physical
operations; duplicates must not count twice. The runner must stop on exhaustion
and bound malformed model outputs and safely rejected choices as well. Preserve
an orderly finish path. No gateway budget semantics are changed by this record.

## KI-02: an interruption can precede delivery of its evidence

`robots/arx/gateway/journal.py:Journal.events()` limits each page to 100 public
records. `EpisodeDriver.run()` fetches one page before invoking the agent.
A long operation can therefore produce an interrupted snapshot whose triggering
proposal is absent from the delivered page.

Reproduced with two 32-action Zeva chunks and a trigger at step 60: the first
100-event page lacked the proposal; the second contained it. Impact: recovery
may be chosen without the critic explanation and supporting observations.

Future resolution: drain pages for a stable snapshot before deciding, explicitly
carry the active trigger, and bound event collection. Journal sequence numbers
also cover private records, so a public sequence need not equal the snapshot
sequence. Do not wait forever for that equality.

## KI-03: existing driver context is not the planned deployment contract

`robots/arx/gateway/episode.py` documents an episode-long conversation and sends
incremental events plus only `last_result`. The agreed orchestration contract
requires a fresh planner per decision and runner-owned bounded explicit history.
A callable could implement this itself, but the driver does not establish it.

Future resolution: the ARX deployment adapter and runner explicitly implement
the existing learning plan's decision input/output schemas and last-16 history.
Do not rely on provider memory or reuse a learner conversation.

## KI-04: standalone startup does not enforce campaign compatibility

`serve_arx_gateway.CoreFactory` registers a package without supplying expected
contract/catalog/bootstrap hashes. `load_candidate()` supports these checks,
but omitted expectations mean only package integrity is checked. This limitation
is already documented in `arx-critic-running.md`.

Future resolution: the harness binds independently computed expected identities
to a frozen trial specification and checks the actual exposed catalog before
motion. A manifest's own declarations are not independent expected identities.
Campaign contract identity and VLA model-contract identity are distinct.

## KI-05: loaded candidates currently have no successful reentry policy

The live factory installs `AlwaysIneligibleReentry`; the supported fixture cannot
mint a reentry token. Recovery actions may be permitted, but recovery-to-Zeva
continuation cannot succeed through that policy. This is an intentional,
documented delivery limitation, not a reason to bypass reentry checks.

Future resolution: separately register a reviewed observation-derived reentry
policy. Until then, report interruption/recovery/finish accurately and do not
claim a complete successful recovery-to-nominal cycle.

## Validation record and unresolved test coverage

Review runs passed 26 gateway tests (including worker timeout/lease tests) and
16 critic tests (including real isolated feature execution). The gateway
HTTP/ASGI test hung and a bounded rerun was terminated; the trace did not establish
a gateway defect. The critic HTTP handoff test was not verified in this review.
Live GPU/MuJoCo execution was not repeated. Resolve this HTTP test coverage gap
before claiming the new runner's end-to-end acceptance tests pass.
