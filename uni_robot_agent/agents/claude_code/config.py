"""Claude-only settings and validation of supported model services."""
import os
import argparse
import math
from urllib.parse import urlsplit


class ClaudeConfig:
    model_protocol = "messages"

    @classmethod
    def add_cli_arguments(cls, parser):
        group = parser.add_argument_group("Claude Code SDK")
        group.add_argument("--claude-max-budget-usd", type=float, default=argparse.SUPPRESS,
                           help="Optional SDK cost ceiling; unset disables the SDK budget cap")
        group.add_argument("--claude-effort", choices=("low", "medium", "high", "max"), default=argparse.SUPPRESS)
        group.add_argument("--claude-cli-path", default=argparse.SUPPRESS)
        group.add_argument("--claude-native-tools", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    @classmethod
    def resolve_defaults(cls, args):
        args.claude_max_budget_usd = getattr(args, "claude_max_budget_usd", None)
        args.claude_effort = getattr(args, "claude_effort", None)
        args.claude_cli_path = getattr(args, "claude_cli_path", None)
        args.claude_native_tools = getattr(args, "claude_native_tools", True)
        if args.reasoning_effort is None:
            args.reasoning_effort = "none"
        if args.provider_url is None:
            from uni_robot_agent.config import runtime_env
            args.provider_url = runtime_env("PROVIDER_URL") or os.environ.get("ANTHROPIC_BASE_URL")

    @classmethod
    def validate_config(cls, args, explicit):
        if "--codex-base-prompt" in explicit:
            raise ValueError("--codex-base-prompt requires --agent codex")
        if explicit & {"--mcp-parallel-dispatch", "--no-mcp-parallel-dispatch", "--mcp-timeout-s"}:
            raise ValueError("Codex MCP dispatch options require --agent codex")
        if args.reasoning_effort != "none":
            raise ValueError("Use --claude-effort for Claude Code reasoning configuration")
        from uni_robot_agent.provider.protocols.messages import validate_context_policy
        validate_context_policy(args.context_policy)
        if not args.provider_url:
            raise ValueError("Claude Code requires --provider-url or ANTHROPIC_BASE_URL")
        if urlsplit(args.provider_url).scheme not in {"http", "https"}:
            raise ValueError("Claude Messages provider URL must use HTTP or HTTPS")
        if args.claude_max_budget_usd is not None and (
            not math.isfinite(args.claude_max_budget_usd) or args.claude_max_budget_usd <= 0
        ):
            raise ValueError("--claude-max-budget-usd must be positive")

    @classmethod
    def config_options(cls, args):
        return {"api_key": args._api_key, "max_budget_usd": args.claude_max_budget_usd,
                "effort": args.claude_effort, "cli_path": args.claude_cli_path,
                "native_tools": args.claude_native_tools}
