def _parts(value, initial=(), floor=0):
    result = list(initial)
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if len(result) <= floor:
                raise ValueError("path escapes scope")
            result.pop()
        else:
            result.append(part)
    return result


def resolve_path(root, path):
    if "\0" in root or "\0" in path or not root.startswith("/"):
        raise ValueError("invalid path")
    base = _parts(root)
    if path.startswith("/"):
        result = _parts(path)
        if result[:len(base)] != base:
            raise ValueError("outside root")
    else:
        result = _parts(path, base, len(base))
    return "/" + "/".join(result)
