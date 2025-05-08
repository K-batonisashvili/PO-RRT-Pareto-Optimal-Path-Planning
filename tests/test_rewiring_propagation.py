import pytest
import numpy as np
from PO_RRT_Star import Node, Path, Tree, Grid, GRID_WIDTH, GRID_HEIGHT, X_MIN, Y_MIN, GRID_RESOLUTION, PARETO_RADIUS
from helper_functions import distance_to
# Need to import the main module to temporarily change PARETO_RADIUS for some tests
import PO_RRT_star

# Create a dummy Grid for Tree initialization and cost calculation using a fixture
@pytest.fixture
def dummy_grid_instance_empty():
    """Provides a basic empty Grid instance for tests."""
    return Grid(GRID_WIDTH, GRID_HEIGHT, [])

# Create a Tree with a simple structure for rewiring tests
@pytest.fixture
def tree_for_rewiring(dummy_grid_instance_empty):
    """Provides a Tree instance with a simple branch for rewiring tests."""
    tree = Tree(dummy_grid_instance_empty)
    root = Node(0, 0, 0)
    tree.add_node(root) # Path 1: [root]

    # Create a branch: root -> node1 -> node2
    node1 = Node(1.0, 0, 0)
    node1.parent = root
    root.children.append(node1)
    node1.cost = distance_to(root, node1)
    node1.log_survival = 0.0
    node1.p_fail = 0.0
    tree.add_node(node1, multiple_children=False) # Path 1: [root, node1]

    node2 = Node(2.0, 0, 0)
    node2.parent = node1
    node1.children.append(node2)
    node2.cost = node1.cost + distance_to(node1, node2)
    node2.log_survival = node1.log_survival
    node2.p_fail = 1 - np.exp(node2.log_survival)
    tree.add_node(node2, multiple_children=False) # Path 1: [root, node1, node2]

    return tree

# Fixture for a new node that could potentially rewire
@pytest.fixture
def new_node_candidate():
    """Provides a new Node instance as a rewiring candidate."""
    new_node = Node(1.5, 0.5, 0)
    # Assume new_node is connected to root for initial cost calculation
    # This cost will be used in the rewire dominance check
    new_node.cost = distance_to(Node(0,0,0), new_node) # Approx 1.58
    new_node.log_survival = 0.0
    new_node.p_fail = 0.0
    return new_node


def test_propagate_cost(tree_for_rewiring, dummy_grid_instance_empty):
    """Checks if propagate_cost correctly updates metrics and path references."""
    tree = tree_for_rewiring
    root = tree.paths[0].nodes[0]
    node1 = root.children[0]
    node2 = node1.children[0]

    # Create a subtree: node2 -> child1 -> grandchild1
    child1 = Node(3.0, 0, 0)
    child1.parent = node2
    node2.children.append(child1)
    # Initial metrics for child1 (will be updated by propagate_cost)
    child1.cost = node2.cost + distance_to(node2, child1)
    child1.log_survival = node2.log_survival
    child1.p_fail = 1 - np.exp(child1.log_survival)
    tree.add_node(child1, multiple_children=False) # Still on Path 1

    grandchild1 = Node(3.5, 0, 0)
    grandchild1.parent = child1
    child1.children.append(grandchild1)
    # Initial metrics for grandchild1 (will be updated by propagate_cost)
    grandchild1.cost = child1.cost + distance_to(child1, grandchild1)
    grandchild1.log_survival = child1.log_survival
    grandchild1.p_fail = 1 - np.exp(grandchild1.log_survival)
    tree.add_node(grandchild1, multiple_children=False) # Still on Path 1

    # Artificially change the cost and log_survival of node2 (the root of propagation)
    node2.cost = 10.0
    node2.log_survival = -np.log(0.5) # 50% survival
    node2.p_fail = 1 - np.exp(node2.log_survival)

    # Propagate from node2
    tree.propagate_cost(node2, dummy_grid_instance_empty.grid)

    # Check child1's updated metrics and path
    expected_child1_cost = node2.cost + distance_to(node2, child1)
    expected_child1_log_survival = node2.log_survival + 0.0 # No grid risk in empty grid
    expected_child1_p_fail = 1 - np.exp(expected_child1_log_survival)
    assert np.isclose(child1.cost, expected_child1_cost)
    assert np.isclose(child1.log_survival, expected_child1_log_survival)
    assert np.isclose(child1.p_fail, expected_child1_p_fail)
    assert child1.path == node2.path # Path reference should be updated

    # Check grandchild1's updated metrics and path
    expected_grandchild1_cost = child1.cost + distance_to(child1, grandchild1)
    expected_grandchild1_log_survival = child1.log_survival + 0.0 # No grid risk in empty grid
    expected_grandchild1_p_fail = 1 - np.exp(expected_grandchild1_log_survival)
    assert np.isclose(grandchild1.cost, expected_grandchild1_cost)
    assert np.isclose(grandchild1.log_survival, expected_grandchild1_log_survival)
    assert np.isclose(grandchild1.p_fail, expected_grandchild1_p_fail)
    assert grandchild1.path == node2.path # Path reference should be updated


def test_rewire_dominating_path(tree_for_rewiring, new_node_candidate, dummy_grid_instance_empty):
    """Checks rewiring when the new path dominates the old one."""
    tree = tree_for_rewiring
    root = tree.paths[0].nodes[0]
    node1 = root.children[0]
    node2 = node1.children[0]
    new_node = new_node_candidate

    # We want new_node -> node2 to be better than node1 -> node2
    # Current path to node2: root -> node1 -> node2 (cost = 2.0, p_fail = 0.0)
    # Potential new path to node2: root -> new_node -> node2
    # Distance from new_node (1.5, 0.5, 0) to node2 (2.0, 0, 0) is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) approx 0.707
    # Cost via new_node = new_node.cost + distance(new_node, node2)
    # Let's manually make the cost via new_node lower for dominance testing
    new_node.cost = 0.5 # Artificially low cost to new_node
    cost_via_new_node = new_node.cost + distance_to(new_node, node2) # 0.5 + 0.707 = 1.207
    p_fail_via_new_node = new_node.p_fail # 0.0

    # Original cost to node2 is 2.0, p_fail is 0.0
    # The new path (cost 1.207, p_fail 0.0) dominates the old path (cost 2.0, p_fail 0.0) because cost is strictly less.

    # Need to add new_node to the tree and link it to root for the dominance check in rewire
    new_node.parent = root
    root.children.append(new_node)
    # Add new_node to a path (could be a new path or existing, doesn't strictly matter for rewire check)
    # For simplicity, let's add it to a new path
    new_path_for_new_node = Path()
    new_path_for_new_node.add_node(root)
    new_path_for_new_node.add_node(new_node)
    tree.paths.append(new_path_for_new_node)
    new_node.path = new_path_for_new_node # Ensure path reference is set

    # Temporarily increase PARETO_RADIUS to include node2 as a neighbor of new_node
    original_pareto_radius = PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = 1.0 # Assuming 1.0 is enough to include node2

    # Find neighbors of new_node with the temporarily increased radius
    znear = tree.neighbors(new_node)
    assert node2 in znear # Ensure node2 is now a neighbor

    # Before rewiring
    assert node2.parent == node1
    assert node2 in node1.children
    assert node2 not in new_node.children
    original_path_node2 = node2.path
    assert node2 in original_path_node2.nodes
    original_rewire_count = tree.rewire_counts

    # Perform rewiring
    tree.rewire(znear, new_node, dummy_grid_instance_empty.grid)

    # After rewiring
    assert node2.parent == new_node
    assert node2 not in node1.children
    assert node2 in new_node.children
    assert node2.path == new_node.path # Path reference should be updated
    assert np.isclose(node2.cost, cost_via_new_node)
    assert np.isclose(node2.p_fail, p_fail_via_new_node)
    assert tree.rewire_counts == original_rewire_count + 1

    # Restore original PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = original_pareto_radius


def test_rewire_no_dominance(tree_for_rewiring, new_node_candidate, dummy_grid_instance_empty):
    """Checks rewiring when the new path does NOT dominate the old one."""
    tree = tree_for_rewiring
    root = tree.paths[0].nodes[0]
    node1 = root.children[0]
    node2 = node1.children[0]
    new_node = new_node_candidate

    # We want new_node -> node2 to NOT be better than node1 -> node2
    # Current path to node2: root -> node1 -> node2 (cost = 2.0, p_fail = 0.0)
    # Potential new path to node2: root -> new_node -> node2
    # We again manually make the cost via new_node higher
    new_node.cost = 3.0 # Artificially high cost to new_node
    cost_via_new_node = new_node.cost + distance_to(new_node, node2) # 3.0 + 0.707 = 3.707
    p_fail_via_new_node = new_node.p_fail # 0.0

    # Original cost to node2 is 2.0, p_fail is 0.0
    # The new path (cost 3.707, p_fail 0.0) does NOT dominate the old path (cost 2.0, p_fail 0.0)

    # Add new_node to the tree and link it to root
    new_node.parent = root
    root.children.append(new_node)
    new_path_for_new_node = Path()
    new_path_for_new_node.add_node(root)
    new_path_for_new_node.add_node(new_node)
    tree.paths.append(new_path_for_new_node)
    new_node.path = new_path_for_new_node # Ensure path reference is set

    # Temporarily increase PARETO_RADIUS to include node2
    original_pareto_radius = PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = 1.0
    znear = tree.neighbors(new_node)
    assert node2 in znear

    # Before rewiring
    assert node2.parent == node1
    assert node2 in node1.children
    assert node2 not in new_node.children
    original_path_node2 = node2.path
    assert node2 in original_path_node2.nodes
    original_rewire_count = tree.rewire_counts

    # Perform rewiring
    tree.rewire(znear, new_node, dummy_grid_instance_empty.grid)

    # After rewiring - no changes should have occurred
    assert node2.parent == node1
    assert node2 in node1.children
    assert node2 not in new_node.children
    assert node2.path == original_path_node2
    assert np.isclose(node2.cost, 2.0) # Should remain the original cost
    assert np.isclose(node2.p_fail, 0.0) # Should remain the original p_fail
    assert tree.rewire_counts == original_rewire_count # No rewiring should happen

    # Restore original PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = original_pareto_radius


def test_rewire_with_grid_risk_propagation(tree_for_rewiring, new_node_candidate):
    """Checks rewiring correctly incorporates grid risk in propagation."""
    tree = tree_for_rewiring
    root = tree.paths[0].nodes[0]
    node1 = root.children[0]
    node2 = node1.children[0]
    new_node = new_node_candidate

    # Create a grid with a risky cell where child1 is located
    risky_obstacles = [{"type": "rectangular", "x_range": (2.9, 3.1), "y_range": (-0.1, 0.1), "probability": 0.5}]
    risky_grid_instance = Grid(GRID_WIDTH, GRID_HEIGHT, risky_obstacles)
    risky_grid_data = risky_grid_instance.grid

    # Create a subtree: node2 -> child1
    child1 = Node(3.0, 0, 0)
    child1.parent = node2
    node2.children.append(child1)
    # Initial metrics for child1 (will be updated by propagate_cost)
    child1.cost = node2.cost + distance_to(node2, child1)
    child1.log_survival = node2.log_survival
    child1.p_fail = 1 - np.exp(child1.log_survival)
    tree.add_node(child1, multiple_children=False) # Still on Path 1

    # Artificially make the new_node path dominate the old path to node2
    new_node.cost = 0.5 # Artificially low cost to new_node

    # Add new_node to the tree and link it to root
    new_node.parent = root
    root.children.append(new_node)
    new_path_for_new_node = Path()
    new_path_for_new_node.add_node(root)
    new_path_for_new_node.add_node(new_node)
    tree.paths.append(new_path_for_new_node)
    new_node.path = new_path_for_new_node # Ensure path reference is set


    # Temporarily increase PARETO_RADIUS
    original_pareto_radius = PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = 1.0
    znear = tree.neighbors(new_node)
    assert node2 in znear

    # Perform rewiring using the risky grid
    tree.rewire(znear, new_node, risky_grid_data)

    # After rewiring, node2's metrics are updated based on new_node
    # Now propagate cost from node2 using the risky grid
    tree.propagate_cost(node2, risky_grid_data)

    # Check child1's updated metrics - should include risk from its cell
    expected_log_s_step_child1 = np.log(1 - 0.5) # Risk at child1's location (3.0, 0)
    expected_child1_cost = node2.cost + distance_to(node2, child1)
    expected_child1_log_survival = node2.log_survival + expected_log_s_step_child1
    expected_child1_p_fail = 1 - np.exp(expected_child1_log_survival)

    assert np.isclose(child1.cost, expected_child1_cost)
    assert np.isclose(child1.log_survival, expected_child1_log_survival)
    assert np.isclose(child1.p_fail, expected_child1_p_fail)
    assert child1.path == node2.path

    # Restore original PARETO_RADIUS
    PO_RRT_star.PARETO_RADIUS = original_pareto_radius


