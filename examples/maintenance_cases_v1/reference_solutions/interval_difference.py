def _union(spans):
    valid = []
    for start, end in spans:
        if type(start) is not int or type(end) is not int or start > end:
            raise ValueError("invalid interval")
        if start < end:
            valid.append((start, end))
    result = []
    for start, end in sorted(valid):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract_intervals(intervals, blocked):
    source, cuts = _union(intervals), _union(blocked)
    result = []
    for start, end in source:
        cursor = start
        for left, right in cuts:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append((cursor, left))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result
