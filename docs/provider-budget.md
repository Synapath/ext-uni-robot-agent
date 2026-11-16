# Persistent provider admission

RUA owns model networking, buffering, cancellation, retries and SDK compatibility.
`provider/budget/` adds optional persistent admission to the existing
`ProviderGateway → ModelClient → ModelTransport` path. It accepts native Responses
or Messages passthrough; conversion adapters are rejected for this policy.
Existing gateways without admission retain their previous behavior.

Experiments supply limits, identities and paths, and may add observation validators.
They must not implement another HTTP transport or retry loop. Sol/Fable profiles
preserve the previously frozen experiment price bounds, including cache TTL usage.
These are not a current vendor price catalog or a provider billing guarantee.
Validate pricing/capacity before authorizing another run.

## Ownership and trust

For isolated SDKs, use two RUA gateways with separate duties:

```text
SDK → worker RUA gateway (context + tool delivery identities)
    → trusted RUA gateway (authenticated admission + private key + SQLite)
    → RUA ModelClient/ModelTransport → vendor API
```

The SDK gets a disposable local token. The experiment's sandbox must deny access
to the vendor key and ledger; admission is not an access-control boundary.
Selected protocol headers are forwarded, authentication is injected privately,
and auxiliary routes are rejected. Count/generation attempts share provider traces.
The ledger stores identities and numeric usage, not prompts or credentials.
Attempt logs store paths, request digests, exception types and OS/TLS codes.

## API and lifecycle

Use `provider.budget.Ledger`, `SolLimits`/`DirectSolLimits` or
`FableLimits`/`DirectFableLimits`, and their native budget gates. Claim the episode
once, construct its gate with `forward=None`, then call
`serve_gate(gate, local_token, key_file=private_path, retry_policy=policy)`.
The result is `(service, endpoint)`. Give that endpoint/token to the worker RUA
runtime with worker provider retries disabled. `service.cancel()` interrupts
sockets/backoff; `service.shutdown()` cancels and drains ASGI workers. Close SQLite
only after shutdown completes, never under an active exchange.
`ProviderConnection` provides direct count/probe access through the same client;
its caller owns admission. Injecting `gate.forward` is for offline fixtures only.

Admission validates model/effort/tools, counts selected input, reserves conservative
input/output cost, generates, checks terminal usage, settles once, then delivers.
Policy hashes, one-shot claims, fingerprints and pending reservations prevent
budget reset or generation replay. Missing usage or ambiguous sending retains
the reservation and stops. Lost downstream delivery also stops. Live simulator
preservation/resume after retry exhaustion is not implemented.

`RetryPolicy(max_retries=3, initial_delay_s=1, backoff_multiplier=2, max_delay_s=8)`
allows three **additional** attempts. Multiplier1 gives fixed delays; N0 disables
retries. Count requests may retry connection/read/write/protocol transport errors.
Generations retry only ConnectError/ConnectTimeout, preserving bytes and the one
reservation. Read/write failures, HTTP statuses and terminal errors are not
replayed. Cancellation and ledger deadlines are checked during backoff.
This mechanism never retries physical tools.

## SDK compatibility and visual context

Native Messages preserve thinking/signatures, tool IDs, cache controls, adaptive
thinking and effort. A custom context policy must declare `wire_formats` containing
`messages`; identity remains the default. `context.messages.project_observations`
projects user evidence and reconstructs messages without rewriting assistant blocks.

`claude_cli_path` selects a CLI and `claude_native_tools=False` excludes native
shell/file tools. SDK retries, automatic memory, tool search, plugins and fallback
are disabled for this controlled route. The pinned CLI MCP output cap is raised
to128000 tokens to preserve observation JSON before context selection. Provider
admission still bounds actual API input; validators must reject any remaining
truncation. `context.image_transport` carries digest-checked SDK placeholders and
restores selected originals before counting/generation. Claude faithful SFT export
remains unsupported; native exchanges and raw tool artifacts are retained.

Codex uses `AgentConfig.codex_sdk_options` (Python API) for restricted runtime,
context and catalog options, owned by `agents/codex/`. Experiments no longer patch
SDK constructors. Do not enable a second SDK retry loop.

## Offline validation

With existing RUA dependencies, from the RUA checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/validation/provider_budget.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/validation/messages_arguments.py -q
```

Synthetic credentials and local HTTP cover both native protocols/authentication,
safe retry phases, single reservation/settlement, deadlines/cancellation, signed
history and unknown-generation retention. SDK mocks prove wire/control compatibility,
not current model access, current billing prices or robot success rates.

Vendor references: [Messages token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
and [thinking/tool workflows](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).
