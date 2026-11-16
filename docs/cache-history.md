# Append-only text and current RGB caching

`context_policy="append_only_observation"` is an optional native Responses
policy. Existing identity/window policies, robot control, scoring and physical
limits keep their defaults. The prepared consumer uses Sol/high; this policy
does not implement Messages/Claude caching.

One ContextManager owns one policy instance per session:

```text
T:   fixed context | durable text history                            | current visual T
T+1: fixed context | durable history + native output T + receipt T    | current visual T+1
```

Raw SDK history only appends. Earlier mutation, compaction, orphan pairs,
changed instructions/model/tool controls or regressing state fails closed.
Each result is projected once and then reused. Native model output order,
visible prose, opaque items and call IDs remain. Repeated prepare is idempotent;
invalid preparation commits no partial state. Historical measured state/time,
localization successes/failures and skills remain. Shared context helpers do
not query hardware, import a concrete environment or generate a task summary.

Calibration never enters persistent receipts. The final CURRENT OBSERVATION
message holds one latest synchronized group, matching calibration and real
source/frame/time labels. Default cameras are head/left_wrist/right_wrist; the
Python constructor accepts another declared tuple. Missing cameras fail rather
than creating frames. Up to six labeled demonstration images and three current
localization overlays are separate. Old raw images and records stay archived.
The English header explains fixed/episode context, historical records, current
pixels versus image references, clocks and full-pose/opening completion.

## Boundaries and admission

The policy sends `prompt_cache_options={"mode":"explicit","ttl":"30m"}`.
Fixed initial context and the last two complete receipt boundaries are marked.
The previous cached endpoint remains searchable as the new endpoint is added;
at most three markers are sent. Marker metadata rolls without rewriting text.
The dynamic suffix is ordinary input, not free. Stable prefixes do not guarantee
hits or shorten context length, and do not create hidden server conversations.

Optional `CacheStableDirectSolBudgetGate` independently verifies the effective
append-only prefix/current view at the trusted hop. Restricted admission only
accepts prepared explicit/30m options: long TTL, prewarm and unprepared markers
fail closed. Counting keeps identical text/images/schema, removing only cache
control metadata. Conservative reservation, actual-usage settlement and safe
retry behavior remain unchanged. No physical action is replayed.

Select via AgentConfig or `--context-policy` with an explicit provider URL.
The consumer must enforce the selected camera contract. Old partial-camera
transcripts are negative replay fixtures, not permission to invent images or
silently rewrite model commands.

## Checks

```sh
PYTHONPATH=. python scripts/validation/cache_history.py
PYTHONPATH=. python scripts/validation/provider_budget.py
```

Checks cover long history, old/new boundaries, idempotence, mutation/compaction,
partial cameras, unknown execution, current frame times, trusted admission and
count/generation semantics. Consumer acceptance adds saved raw SDK replay and
the real pinned Codex SDK with original-sized PNGs, a local fake provider and
connection-failure injection. Synthetic usage is not a real hit rate or fee.
Actual vendor schema acceptance, cache behavior and robot success require a
separately authorized live experiment. No simulator resume is implemented.

Reference: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
