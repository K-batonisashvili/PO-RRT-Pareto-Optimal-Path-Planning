import pytest
import numpy as np
from PO_RRT_Star import Node, Tree, Grid, GRID_WIDTH, GRID_HEIGHT, X_MIN, Y_MIN, GRID_RESOLUTION, PARETO_RADIUS
from helper_functions import distance_to

# Create a dummy Grid for Tree initialization and cost calculation using a fixture
@pytest.fixture
def dummy_grid_instance_empty():
    """Provides a basic empty Grid instance for tests."""
    return Grid(GRID_WIDTH, GRID_HEIGHT, [])

# Create a Tree with a root node using a fixture
@pytest.fixture
def tree_with_root(dummy_grid_instance_empty):
    """Provides a Tree instance with a root node."""
    tree = Tree(dummy_grid_instance_empty)
    root = Node(0, 0, 0)
    tree.add_node(root) # Add root to the tree
    return tree

# Fixture to create a set of potential parent nodes (znear)
@pytest.fixture
def potential_parents(tree_with_root):
    """Provides a list of potential parent nodes linked to the root."""
    root = tree_with_root.paths[0].nodes[0]

    parent1 = Node(1.0, 0, 0)
    parent1.parent = root # Link to root for path tracing if needed
    root.children.append(parent1)
    parent1.cost = distance_to(root, parent1)
    parent1.log_survival = 0.0 # Assume no risk on path to parent1
    parent1.p_fail = 0.0
    tree_with_root.add_node(parent1, multiple_children=False) # Add to tree

    parent2 = Node(0, 1.0, 0)
    parent2.parent = root
    root.children.append(parent2)
    parent2.cost = distance_to(root, parent2)
    parent2.log_survival = 0.0
    parent2.p_fail = 0.0
    tree_with_root.add_node(parent2, multiple_children=True) # Fork for parent2

    return [parent1, parent2]

def test_choose_parents_single_pareto_optimal(tree_with_root, potential_parents, dummy_grid_instance_empty):
    """Checks choose_parents when only one parent is Pareto-optimal."""
    tree = tree_with_root
    znear = potential_parents
    new_node_x, new_node_y, new_node_theta = 1.0, 1.0, 0

    # Assume parent1 offers a better path to (1,1,0) than parent2
    # Artificially set costs/log_survival to ensure this dominance for the path to (1,1,0)
    # Path from root -> parent1 -> (1,1,0)
    cost_p1_to_new = distance_to(znear[0], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p1 = znear[0].cost + cost_p1_to_new
    total_log_survival_p1 = znear[0].log_survival  # No additional risk for parent1

    # Path from root -> parent2 -> (1,1,0)
    cost_p2_to_new = distance_to(znear[1], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p2 = znear[1].cost + cost_p2_to_new
    total_log_survival_p2 = znear[1].log_survival + np.log(1 - 0.1)  # Add risk for parent2

    znear[0].cost = total_cost_p1
    znear[0].log_survival = total_log_survival_p1
    znear[1].cost = total_cost_p2
    znear[1].log_survival = total_log_survival_p2

    new_nodes = tree.choose_parents(
        znear,
        new_node_x, new_node_y, new_node_theta,
        dummy_grid_instance_empty.grid  # Pass grid data for risk calculation
    )

    assert len(new_nodes) == 1  # Only one Pareto-optimal parent should be chosen
    chosen_node = new_nodes[0]
    assert chosen_node.parent == znear[0]  # Parent1 should be the chosen parent
    # Check the cost and log_survival of the new node (calculated based on the chosen parent)
    assert np.isclose(chosen_node.cost, znear[0].cost + cost_p1_to_new)
    assert np.isclose(chosen_node.log_survival, znear[0].log_survival)


def test_choose_parents_multiple_pareto_optimal(tree_with_root, potential_parents, dummy_grid_instance_empty):
    """Checks choose_parents when multiple parents are Pareto-optimal (non-dominated)."""
    tree = tree_with_root
    znear = potential_parents
    new_node_x, new_node_y, new_node_theta = 1.0, 1.0, 0

    # Assume both parent1 and parent2 offer non-dominated paths to (1,1,0)
    # Let's set costs/log_survival such that neither dominates the other
    # Path from root -> parent1 -> (1,1,0)
    cost_p1_to_new = distance_to(znear[0], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p1 = znear[0].cost + cost_p1_to_new
    total_log_survival_p1 = znear[0].log_survival + np.log(1 - 0.1)  # Higher risk

    # Path from root -> parent2 -> (1,1,0)
    cost_p2_to_new = distance_to(znear[1], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p2 = znear[1].cost + cost_p2_to_new + 0.5  # Higher cost
    total_log_survival_p2 = znear[1].log_survival + np.log(1 - 0.05)  # Lower risk

    znear[0].cost = total_cost_p1
    znear[0].log_survival = total_log_survival_p1
    znear[1].cost = total_cost_p2
    znear[1].log_survival = total_log_survival_p2

    new_nodes = tree.choose_parents(
        znear,
        new_node_x, new_node_y, new_node_theta,
        dummy_grid_instance_empty.grid
    )

    assert len(new_nodes) == 2  # Both should be returned as they are non-dominated

    # Check that the returned nodes correspond to the correct parents and metrics
    parents = [n.parent for n in new_nodes]
    assert znear[0] in parents
    assert znear[1] in parents

    for node in new_nodes:
        if node.parent == znear[0]:
            assert np.isclose(node.cost, znear[0].cost + cost_p1_to_new)
            assert np.isclose(node.log_survival, znear[0].log_survival)
        elif node.parent == znear[1]:
            assert np.isclose(node.cost, znear[1].cost + cost_p2_to_new)
            assert np.isclose(node.log_survival, znear[1].log_survival)
        else:
            pytest.fail("Unexpected parent in returned nodes")


def test_choose_parents_dominated_candidates_filtered(tree_with_root, potential_parents, dummy_grid_instance_empty):
    """Checks choose_parents filters out dominated candidates."""
    tree = tree_with_root
    znear = potential_parents
    new_node_x, new_node_y, new_node_theta = 1.0, 1.0, 0

    # Parent1 dominates Parent2 for the path to (1,1,0)
    # Path from root -> parent1 -> (1,1,0)
    cost_p1_to_new = distance_to(znear[0], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p1 = znear[0].cost + cost_p1_to_new
    total_log_survival_p1 = znear[0].log_survival + np.log(1 - 0.1)

    # Path from root -> parent2 -> (1,1,0) - dominated by path from parent1
    cost_p2_to_new = distance_to(znear[1], Node(new_node_x, new_node_y, new_node_theta))
    total_cost_p2 = znear[1].cost + cost_p2_to_new + 0.5  # Higher cost
    total_log_survival_p2 = znear[1].log_survival + np.log(1 - 0.2)  # Higher risk

    znear[0].cost = total_cost_p1
    znear[0].log_survival = total_log_survival_p1
    znear[1].cost = total_cost_p2
    znear[1].log_survival = total_log_survival_p2

    new_nodes = tree.choose_parents(
        znear,
        new_node_x, new_node_y, new_node_theta,
        dummy_grid_instance_empty.grid
    )

    assert len(new_nodes) == 1  # Only the non-dominated one should be returned
    assert new_nodes[0].parent == znear[0]  # Parent1 should be parent
    assert np.isclose(new_nodes[0].cost, znear[0].cost + cost_p1_to_new)
    assert np.isclose(new_nodes[0].log_survival, znear[0].log_survival)


def test_choose_parents_with_grid_risk(tree_with_root, potential_parents):
    """Checks choose_parents correctly incorporates grid risk."""
    tree = tree_with_root
    znear = potential_parents
    new_node_x, new_node_y, new_node_theta = 1.0, 1.0, 0

    # Create a grid with a risky cell at (1,1)
    risky_obstacles = [{"type": "rectangular", "x_range": (0.9, 1.1), "y_range": (0.9, 1.1), "probability": 0.5}]
    risky_grid_instance = Grid(GRID_WIDTH, GRID_HEIGHT, risky_obstacles)
    risky_grid_data = risky_grid_instance.grid

    # Parent1 and Parent2 have no risk on their paths to (1,1,0) initially
    # Temporarily modify parent log_survival/p_fail for the calculation within choose_parents
    original_p1_log_s = znear[0].log_survival
    original_p1_p_fail = znear[0].p_fail
    original_p2_log_s = znear[1].log_survival
    original_p2_p_fail = znear[1].p_fail

    znear[0].log_survival = 0.0
    znear[0].p_fail = 0.0
    znear[1].log_survival = 0.0
    znear[1].p_fail = 0.0

    new_nodes = tree.choose_parents(
        znear,
        new_node_x, new_node_y, new_node_theta,
        risky_grid_data # Use the risky grid data
    )

    # Restore original log_survival/p_fail
    znear[0].log_survival = original_p1_log_s
    znear[0].p_fail = original_p1_p_fail
    znear[1].log_survival = original_p2_log_s
    znear[1].p_fail = original_p2_p_fail

    # The risk in the cell at (1,1) should be added to the log_survival
    expected_log_s_step = np.log(1 - 0.5) # Probability 0.5 at (1,1)
    expected_p_fail_step = 1 - np.exp(expected_log_s_step)

    assert len(new_nodes) == 2 # Assuming initial costs/p_fail are equal for parents and no grid risk on parent paths

    for node in new_nodes:
        assert np.isclose(node.log_survival, node.parent.log_survival + expected_log_s_step)
        assert np.isclose(node.p_fail, 1 - np.exp(node.log_survival)) # Check p_fail calculation
