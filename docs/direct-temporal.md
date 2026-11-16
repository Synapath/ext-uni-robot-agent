# Temporal RGB and bounded Direct motion

These options are opt-in. Existing RoboDojo deployments retain single-snapshot
observations and their existing motion limits. They do not change the task,
native scorer, robot geometry or controller rate.

## Observation contract

Deploy a RoboDojo environment with `rgb_history_seconds=2.0` and
`rgb_history_hz=5.0`. Equivalent environment-server flags are
`--rgb-history-seconds 2 --rgb-history-hz 5`.

The adapter retains real RGB after acknowledged native control steps. Sampling
uses **simulation time**, not host time. At the native 25 Hz rate, the latest
half-open window `(now - 2, now]` contains at most ten synchronized groups from
the head and both wrist cameras. The newest frame is always included. A short
episode supplies only available frames; observation does not advance physics,
repeat frames or pad missing history. Non-divisor sample rates use real control
ticks spaced at least `1 / sample_hz` apart, so inspect actual timestamps rather
than assuming a perfect cadence.

Shared scene presentation attaches an English label immediately before each
image, with camera, frame ID, image index, simulation timestamp, UTC host capture
timestamp and native state version. The UTC timestamp is the host readout time,
not a hardware exposure timestamp. Groups are oldest to newest. Latest images
can use the fresh calibrated capture; calibration and localization always refer
to the latest image, never a historical wrist pose. `temporal_rgb` metadata records
the same ordering and provenance without embedding image bytes in JSON.

For a planner that must see at most one current clip, select
`context_policy="observation_window"`, `context_observation_horizon=1`. Older
clips and their per-frame labels are removed from effective model requests;
action results, failures and text history remain. Duplicate references to one
clip retain only the latest occurrence. Original artifacts remain available to
the recorder. This policy does not summarize the task or promise cache hits.

The planner SDK must preserve the tool's structured observation JSON. Its own
tool-output truncation can corrupt JSON before this policy sees it, preventing
image ownership detection. Set an adequate per-run tool-output allowance in the
SDK integration and reject undecodable temporal observations before generation.
Labels can carry a runtime timing prefix; eviction removes the entire labeled
text block together with its image. A short fixture that omits real-sized JSON
does not validate this boundary.

## Motion contract

`direct_sim_seconds=1.0` advertises an optional shared gripper budget. Schemas
and handlers reject requests beyond that duration. At 25 Hz, existing move and
gripper caps remain 24 and 12 steps; hold is tightened to 25. No command is
silently rewritten. These bounds cover simulation advancement, not model, IK,
rendering or network latency. Stepwise execution is required.

`direct_wall_seconds=1.0` additionally enables a **cooperative** motion-loop
deadline. It waits for an in-flight primitive's acknowledgement, then declines
to submit another primitive. Feedback reports `stop_reason="wall_time_budget"`,
`truncated_by_budget`, executed steps, measured state and an English continuation
note. The caller must inspect partial execution before replanning. An unknown
acknowledgement is not permission to replay a command. This is not a hard
one-second preemption or an emergency stop; an individual primitive, final
capture and response serialization can exceed that deadline. Leave the option
unset when a host's primitive latency makes it unsuitable.

The experiment harness owns the stricter rule of one state-changing tool per
model exchange and the English episode prompt. Use `tool_profile="direct"`,
`interaction_mode="tool_calling"`, `tool_return_mode="immediate"` for a
synchronous feedback boundary without model-generated job waits. The generic
RUA scheduler's background mode remains available for separate experiments.

## Source selection and validation

`DeploymentConfig.rua_source_root` explicitly selects a checked-out package for
the environment server while reusing a prepared native interpreter and assets.
It does not install packages. The planner process must separately select the
same source and pin its commit and hashes. Omit this development option for the
installed-package path; a moving checkout is not a reproducible run identity.

Run `python scripts/validation/direct_temporal.py` for CPU checks of sampling,
labels, clip eviction, motion limits, acknowledged truncation and default
compatibility. Run `python scripts/validation/tool_composition.py` for profile
composition regressions. These checks do not establish native task success or
real-robot safety. Codex terminal SDK errors are reported as errors rather than
clean early completion, including when no robot tool was delivered.
