from copy import deepcopy

DELETE = object()


def merge_config(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        if value is DELETE:
            result.pop(key, None)
        elif isinstance(value, dict):
            previous = result.get(key)
            result[key] = merge_config(previous if isinstance(previous, dict) else {}, value)
        else:
            result[key] = deepcopy(value)
    return result
