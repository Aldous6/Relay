def parse_records(chunks, delimiter=","):
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '\"\r\n':
        raise ValueError("invalid delimiter")
    text = "".join(chunks)
    records, record, field = [], [], []
    state = "start"
    active = False
    i = 0
    while i < len(text):
        ch = text[i]
        if state == "quoted":
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                state = "closed"
            else:
                field.append(ch)
        elif ch == delimiter:
            record.append("".join(field))
            field, state = [], "start"
            active = True
        elif ch in "\r\n":
            if ch == "\r":
                if i + 1 >= len(text) or text[i + 1] != "\n":
                    raise ValueError("bare CR")
                i += 1
            record.append("".join(field))
            records.append(record)
            record, field, state, active = [], [], "start", False
        elif state == "closed":
            raise ValueError("text after quote")
        elif ch == '"':
            if state != "start":
                raise ValueError("quote in plain field")
            state, active = "quoted", True
        else:
            field.append(ch)
            state, active = "plain", True
        i += 1
    if state == "quoted":
        raise ValueError("unclosed quote")
    if active or record or field:
        records.append(record + ["".join(field)])
    return records
