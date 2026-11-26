"""Plan prerequisite-first work for a dependency graph."""


def _visit(node, graph, visited, result):
    if node in visited:
        return
    visited.add(node)
    for dependency in graph.get(node, ()):
        _visit(dependency, graph, visited, result)
    result.append(node)


def _all_nodes(graph):
    nodes = set(graph)
    for dependencies in graph.values():
        for node in dependencies:
            nodes.add(node)
    return nodes


def dependency_order(graph):
    """Visit dependencies before the task that needs them.

    Node names provide deterministic ordering for reproducible plans.
    The traversal supports nodes that only appear as prerequisites.
    Each node is appended once so shared prerequisites are not repeated.
    """
    result = []
    visited = set()
    nodes = _all_nodes(graph)
    for node in sorted(nodes):
        _visit(node, graph, visited, result)
    return result
