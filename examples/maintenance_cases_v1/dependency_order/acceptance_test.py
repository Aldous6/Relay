import copy
import itertools
import unittest
from module import dependency_order


class AcceptanceTests(unittest.TestCase):
    def test_duplicate_edges_missing_nodes_generators(self):
        self.assertEqual(dependency_order({"d": iter(["b", "a", "b"]), "c": iter(["b"])}), ["a", "b", "c", "d"])

    def test_cycles(self):
        for graph in [{"x": ["x"]}, {"a": ["b"], "b": ["a"]}, {"z": [], "x": ["y"], "y": ["x"]}]:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                dependency_order(graph)

    def test_no_mutation_and_dynamic_ready(self):
        graph = {"z": [], "b": ["c"], "a": ["c"], "c": []}
        before = copy.deepcopy(graph)
        self.assertEqual(dependency_order(graph), ["c", "a", "b", "z"])
        self.assertEqual(graph, before)

    def test_small_dags_against_all_valid_orders(self):
        nodes = ["a", "b", "c", "d"]
        edges = [(nodes[i], nodes[j]) for i in range(4) for j in range(i)]
        for mask in range(1 << len(edges)):
            graph = {node: [] for node in nodes}
            for i, (node, dependency) in enumerate(edges):
                if mask & (1 << i):
                    graph[node].append(dependency)
            orders = [p for p in itertools.permutations(nodes)
                      if all(p.index(d) < p.index(n) for n, ds in graph.items() for d in ds)]
            self.assertEqual(dependency_order(graph), list(min(orders)))
