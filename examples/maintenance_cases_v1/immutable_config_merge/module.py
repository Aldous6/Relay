"""Layer application overrides onto a base configuration."""

DELETE = object()


def _copy_container(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _merge_values(previous, incoming):
    if isinstance(previous, dict) and isinstance(incoming, dict):
        return merge_config(previous, incoming)
    return _copy_container(incoming)


def merge_config(base, patch):
    """Apply explicitly supplied overrides to a copied top-level dict.

    A missing override retains the existing value.
    Dictionaries combine their individual keys while lists are replaced.
    This keeps configuration assembly convenient for the caller.
    """
    result = dict(base)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif key in result:
            result[key] = _merge_values(result[key], value)
        else:
            result[key] = _copy_container(value)
    return result
