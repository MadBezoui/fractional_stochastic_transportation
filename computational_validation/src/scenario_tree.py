import pandas as pd
import numpy as np

class Node:
    def __init__(self, node_id, t, parent=None, prob_cond=1.0, demand=None):
        self.node_id = node_id
        self.t = t
        self.parent = parent
        self.children = []
        self.prob_cond = prob_cond
        self.demand = demand or {}  # dict mapping customer j to demand float
        
        if parent is not None:
            parent.children.append(self)
            
    def get_path_to_root(self):
        path = [self]
        curr = self.parent
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

class ScenarioTree:
    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.leaves = []
        self.scenarios = []  # List of dicts, each representing a scenario path
        self._build_node_lists(root)
        self._build_scenarios()
        
    def _build_node_lists(self, node):
        self.nodes.append(node)
        if not node.children:
            self.leaves.append(node)
        for child in node.children:
            self._build_node_lists(child)
            
    def _build_scenarios(self):
        for idx, leaf in enumerate(self.leaves):
            path = leaf.get_path_to_root()
            prob = np.prod([n.prob_cond for n in path])
            self.scenarios.append({
                'omega': idx,
                'path': path,
                'prob': prob
            })

def generate_toy_tree(N=2, J=[1, 2], base_demands={1: 10.0, 2: 15.0}):
    """
    Generates a deterministic toy scenario tree with binary branching per period.
    N: number of periods
    J: list of customers
    base_demands: base demand for each customer
    Returns a ScenarioTree object.
    """
    node_counter = 0
    
    # Root node at t=0, demand is known
    root = Node(node_id=node_counter, t=0, demand={j: base_demands[j] for j in J})
    node_counter += 1
    
    current_level = [root]
    
    for t in range(1, N + 1):
        next_level = []
        for parent in current_level:
            # Branch 1: High demand (+50%)
            d_high = {j: parent.demand[j] * 1.5 for j in J}
            n_high = Node(node_id=node_counter, t=t, parent=parent, prob_cond=0.5, demand=d_high)
            node_counter += 1
            
            # Branch 2: Low demand (-50%)
            d_low = {j: parent.demand[j] * 0.5 for j in J}
            n_low = Node(node_id=node_counter, t=t, parent=parent, prob_cond=0.5, demand=d_low)
            node_counter += 1
            
            next_level.extend([n_high, n_low])
        current_level = next_level
        
    root.demand = {}
    return ScenarioTree(root)

if __name__ == "__main__":
    tree = generate_toy_tree(N=3)
    print(f"Total nodes: {len(tree.nodes)}")
    print(f"Total scenarios: {len(tree.scenarios)}")
    for s in tree.scenarios:
        print(f"Scenario {s['omega']} (prob {s['prob']:.3f}): Nodes {[n.node_id for n in s['path']]}")
