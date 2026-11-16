"""Explicit restricted SDK/context configuration, owned by the Codex adapter."""
import json
from pathlib import Path

def sdk_overrides(provider='uni_robot_agent_proxy', *, tool_output_token_limit=None,
                  model_context_window=None, model_auto_compact_token_limit=None, model_catalog_json=None):
    """Disable ambient integrations, subagents and provider retries in pinned SDK."""
    disabled = ('multi_agent', 'multi_agent_v2', 'goals', 'apps', 'plugins', 'remote_plugin',
                'tool_suggest', 'skill_search', 'skill_mcp_dependency_install', 'browser_use',
                'browser_use_external', 'computer_use', 'image_generation', 'in_app_browser',
                'memories', 'hooks', 'workspace_dependencies', 'fast_mode')
    overrides = [*(f'features.{name}=false' for name in disabled),
            f'model_providers.{provider}.request_max_retries=0',
            f'model_providers.{provider}.stream_max_retries=0']
    for name,value in [('tool_output_token_limit',tool_output_token_limit),
                       ('model_context_window',model_context_window),
                       ('model_auto_compact_token_limit',model_auto_compact_token_limit)]:
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError('Invalid Codex context limit')
            overrides.append(f'{name}={value}')
    if model_catalog_json is not None:
        overrides.append('model_catalog_json='+json.dumps(str(Path(model_catalog_json).resolve(strict=True))))
    return overrides


def bind_sdk_policy(*, tool_output_token_limit=None, model_context_window=None,
                    model_auto_compact_token_limit=None, model_catalog_json=None):
    """Thin adapter to fixed RUA 0.2.0; never writes its checkout or wheel."""
    from uni_robot_agent.agents.codex import agent
    original = agent._sdk_config_overrides
    def overrides(*args, **kwargs):
        return original(*args, **kwargs) + sdk_overrides(tool_output_token_limit=tool_output_token_limit,
            model_context_window=model_context_window,model_auto_compact_token_limit=model_auto_compact_token_limit,
            model_catalog_json=model_catalog_json)
    agent._sdk_config_overrides = overrides

