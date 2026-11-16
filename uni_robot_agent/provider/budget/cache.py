"""Restricted Responses cache controls and semantic token-count projection."""
from .core import Stop


def validate_cache(body):
    if not isinstance(body, dict):
        raise Stop('invalid_cache_request')
    markers = []
    def visit(value):
        if isinstance(value, dict):
            if 'prompt_cache_breakpoint' in value:
                if value.get('type') != 'input_text' or value['prompt_cache_breakpoint'] != {'mode': 'explicit'}:
                    raise Stop('unprepared_cache_breakpoint')
                markers.append(value)
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(body.get('input', []))
    options = body.get('prompt_cache_options')
    if options is None and not markers: return
    if (body.get('model') != 'gpt-6.1-sol' or options != {'mode': 'explicit', 'ttl': '30m'}
            or not 1 <= len(markers) <= 3):
        raise Stop('unprepared_cache_policy')


def count_semantics(value):
    """Cache metadata is generation-only; never remove text, images or schemas."""
    if isinstance(value, dict):
        return {key: count_semantics(child) for key, child in value.items() if key != 'prompt_cache_breakpoint'}
    if isinstance(value, list): return [count_semantics(child) for child in value]
    return value
