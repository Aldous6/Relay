import heapq


def dependency_order(graph):
    dependencies = {node: set(edges) for node, edges in graph.items()}
    nodes = set(dependencies)
    for edges in dependencies.values():
        nodes.update(edges)
    following = {node: [] for node in nodes}
    degree = {node: len(dependencies.get(node, ())) for node in nodes}
    for node, edges in dependencies.items():
        for edge in edges:
            following[edge].append(node)
    ready = [node for node in nodes if degree[node] == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for target in following[node]:
            degree[target] -= 1
            if degree[target] == 0:
                heapq.heappush(ready, target)
    if len(result) != len(nodes):
        raise ValueError("cyclic graph")
    return result
