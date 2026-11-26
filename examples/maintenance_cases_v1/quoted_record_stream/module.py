"""Read delimiter-separated records from text chunks."""


def _fields(line, delimiter):
    fields = []
    current = []
    quoted = False
    for character in line:
        if character == '"':
            quoted = not quoted
        elif character == delimiter and not quoted:
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if quoted:
        raise ValueError("unclosed quote")
    fields.append("".join(current))
    return fields


def parse_records(chunks, delimiter=","):
    """Buffer partial lines and emit completed records."""
    result = []
    pending = ""
    for chunk in chunks:
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            result.append(_fields(line.rstrip("\r"), delimiter))
    if pending:
        result.append(_fields(pending, delimiter))
    return result
