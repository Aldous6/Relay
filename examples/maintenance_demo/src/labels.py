"""Small maintenance target with an intentional normalization bug."""


def normalize_labels(labels):
    """Return nonempty normalized labels, without duplicates, in input order."""
    return list(set(labels))
