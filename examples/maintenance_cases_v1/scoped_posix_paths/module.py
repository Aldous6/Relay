"""Resolve virtual resource names within a configured namespace."""


def _collapse(value):
    parts = []
    for component in value.split("/"):
        if component in ("", "."):
            continue
        if component == "..":
            if parts:
                parts.pop()
            continue
        parts.append(component)
    return "/" + "/".join(parts)


def _join(root, path):
    if path.startswith("/"):
        return path
    return root.rstrip("/") + "/" + path


def resolve_path(root, path):
    """Canonicalize the resource location and check its root prefix."""
    if not root.startswith("/"):
        raise ValueError("absolute root required")
    normalized_root = _collapse(root)
    candidate = _join(normalized_root, path)
    resolved = _collapse(candidate)
    if not resolved.startswith(normalized_root):
        raise ValueError("path outside root")
    return resolved
