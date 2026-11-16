# Native termination and planner admission

An acknowledged terminal state ends planning independently of a model's `finish`
claim and its request allowance. Prepared sessions bind an internal terminal
callback to the toolkit. Uncertain execution is not a terminal acknowledgement.

The SDK normally requests another response after returning a tool result. The
RUA gateway retains that final context and the last tool receipt, but checks the
terminal callback before uploads, upstream counting or generation. It returns an
explicit local lifecycle stop and cancels the SDK after delivering that control
reply. This is not a synthetic model answer, vendor usage or task verdict.
Responses and Messages gateways support this boundary; the Codex adapter also
understands the resulting expected cancellation. Raw records distinguish local
stops from real model exchanges.

When an allowance is exhausted without termination, trusted admission rejects
before counting and returns a structured admission stop. A loopback consumer
preserves this classification instead of reporting a generic502. Unknown usage
and physical execution retain their reservations; they are not ordinary limits.

Result JSON keeps confirmed native success/failure, native score and completion
alongside provider and cleanup errors. A late SDK failure cannot erase a native
verdict, but can still cause a nonzero engineering exit. Uncertain physical
execution takes precedence over a previous native outcome.

```sh
PYTHONPATH=. python scripts/validation/terminal_lifecycle.py
PYTHONPATH=. python scripts/validation/provider_budget.py
```

Consumer acceptance adds real SDK/local fake provider cases: termination on the
last permitted generation, exhausted allowance without termination, an80-call
boundary and transport faults. These are lifecycle fixtures, not robot success
rates or real vendor measurements. They do not recover past signal senders.
