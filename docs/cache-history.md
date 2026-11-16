# Append-only text and current RGB caching

`context_policy="append_only_observation"` is an optional native Responses
policy, version2. It requires an explicit integration-declared robot profile.
Existing identity/window policies, robot control, scoring and physical
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
source/frame/time labels. The selected profile declares the complete camera set;
there is no implicit X5 robot or25Hz default in shared projection. Missing cameras fail rather
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

## Prepared robot profile and migration

The only integration-declared profile currently prepared is
`robodojo_x5_25hz_v1`: RoboDojo, left/right X5 arms, `x5_grasp_site`, world-frame
metres, fixed XYZ radians, measured opening in metres,25Hz and
head/left_wrist/right_wrist. This does not prepare Piper-X or another robot,
enable depth/object truth, or add Messages/Claude caching. Sol explicit30m
validation remains a separate provider/billing contract.

```python
from uni_robot_agent import AgentConfig
from uni_robot_agent.integrations.robodojo.context_profile import ROBODOJO_X5_25HZ
from uni_robot_agent.context.policies.append_only_observation import AppendOnlyObservationPolicy
from uni_robot_agent.provider.budget.responses import CacheStableDirectSolBudgetGate

agent = AgentConfig(model="gpt-6.1-sol", reasoning_effort="high",
    provider_url="http://127.0.0.1:18000/v1",
    context_policy="append_only_observation",
    context_robot_profile=ROBODOJO_X5_25HZ.name)
policy = AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ)
# Pass profile=ROBODOJO_X5_25HZ when constructing the trusted cache gate too.
```

CLI selection adds `--context-robot-profile robodojo_x5_25hz_v1` with
`--integration robodojo --context-policy append_only_observation`.
This setting is independent of the existing observer's `--observation-profile`.
Missing/unknown robot profiles or another integration's profile fail during
configuration. After connecting, Session validates EpisodeInfo integration,
control rate, arm control frames and calibrated camera capabilities before
SDK/model startup. Invalid startup still records failure and closes resources.

`IntegrationSpec.context_profiles` supplies immutable `RobotContextProfile`
contracts. The concrete X5 declaration lives in `integrations/robodojo/` and
imports no native simulator. Shared `context/` helpers receive rate/arms/site
parameters; they never import RoboDojo or infer a robot from its image names.
The selected profile is included in the fixed English header, compact states
and current-observation metadata. Trusted admission rejects inconsistent
profile/rate/state/current-frame declarations before counting or generation.

Version1 protocols and source pins are historical and remain unchanged; replay
them with their frozen source. New protocols must select the profile and pin the
new source. Direct helper callers must now pass `profile=...`; clock helpers
require `control_hz=...`. There is no silent compatibility fallback to X5.

## Native Messages

The explicit profile also supports native Messages via the same
`append_only_observation` policy. User observations reuse the shared projection;
assistant text, thinking signatures, tool IDs and inline system messages stay
native. Equivalent SDK string/list text forms are canonicalized; previously
recorded images pruned by the CLI are restored from the frozen local history,
then projected again. Other history/header changes fail before count/generation. The
fixed task and last two tool receipts use explicit ephemeral5m cache controls;
the current RGB/calibration tail is never marked. Hits refresh this lifetime.
Responses30m controls are not translated onto the Messages wire. Native count
requests remove only cache metadata, retaining identical text/images/schema.
Use `CacheStableDirectFableBudgetGate(..., profile=...)` for independent trusted
validation. Existing identity/observation-window behavior is unchanged.

```bash
PYTHONPATH=. python scripts/validation/messages_cache_history.py
```

See the [official caching contract](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## Checks

```sh
PYTHONPATH=. python scripts/validation/cache_history.py
PYTHONPATH=. python scripts/validation/provider_budget.py
PYTHONPATH=. python scripts/validation/context_profiles.py
```

Checks cover long history, old/new boundaries, idempotence, mutation/compaction,
partial cameras, unknown execution, current frame times, trusted admission and
count/generation semantics. Consumer acceptance adds saved raw SDK replay and
the real pinned Codex SDK with original-sized PNGs, a local fake provider and
connection-failure injection. Synthetic usage is not a real hit rate or fee.
Actual vendor schema acceptance, cache behavior and robot success require a
separately authorized live experiment. No simulator resume is implemented.

Reference: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
