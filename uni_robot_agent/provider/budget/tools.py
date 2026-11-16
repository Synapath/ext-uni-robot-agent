"""Tool allowlists used by the optional restricted robot provider policy."""
DIRECT_TOOLS = frozenset({'move','grasp','release','hold','capture_scene','view_env_state',
                          'locate','list_skills','read_skill','finish'})
PHYSICAL_TOOLS = frozenset({'move','grasp','release','hold','finish'})
SDK_NATIVE_TOOLS = frozenset({'exec_command','write_stdin','list_mcp_resources',
    'list_mcp_resource_templates','read_mcp_resource','update_plan','request_user_input','view_image'})
