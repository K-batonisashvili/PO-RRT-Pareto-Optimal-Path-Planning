import pytest
import numpy as np
from PO_RRT_star_occupancy import Node, Path, Tree, Grid, GRID_WIDTH, GRID_HEIGHT
from helper_functions import distance_to

# Create a dummy Grid for Tree initialization using a fixture
@pytest.fixture
def dummy_grid_instance():
    """Provides a basic empty Grid instance for tests."""
    return Grid(GRID_WIDTH, GRID_HEIGHT, [])

# Create a Tree with a root node using a fixture
@pytest.fixture
def tree_with_root(dummy_grid_instance):
    """Provides a Tree instance with a root node."""
    tree = Tree(dummy_grid_instance)
    root_node = Node(0, 0, 0)
    tree.add_node(root_node)
    return tree

def test_path_properties():
    """Checks Path cost and p_fail properties."""
    path = Path()
    node1 = Node(0, 0, 0)
    node1.cost = 1.0
    node1.p_fail = 0.1
    path.add_node(node1)
    assert np.isclose(path.cost, 1.0)
    assert np.isclose(path.p_fail, 0.1)

    node2 = Node(1, 0, 0)
    node2.cost = 2.5
    node2.p_fail = 0.2
    path.add_node(node2)
    assert np.isclose(path.cost, 2.5)
    assert np.isclose(path.p_fail, 0.2)

def test_tree_add_node_root(dummy_grid_instance):
    """Checks adding the root node to the tree."""
    tree = Tree(dummy_grid_instance)
    root_node = Node(0, 0, 0)
    tree.add_node(root_node)

    assert len(tree.paths) == 1
    root_path = tree.paths[0]
    assert len(root_path.nodes) == 1
    assert root_path.nodes[0] == root_node
    assert root_node.parent is None
    assert root_node.path == root_path
    assert root_node.added_to_tree is True
    assert tree.node_count == 1

def test_tree_add_node_extend_path(tree_with_root):
    """Checks adding a node that extends an existing path."""
    tree = tree_with_root
    root_node = tree.paths[0].nodes[0] # Get the root node from the tree

    node1 = Node(1, 0, 0)
    node1.parent = root_node
    root_node.children.append(node1)
    node1.cost = distance_to(root_node, node1) # Calculate cost
    node1.log_survival = 0.0
    node1.p_fail = 0.0

    # Add node1, should extend the root's path
    tree.add_node(node1, multiple_children=False)

    assert len(tree.paths) == 1 # Still only one path
    existing_path = tree.paths[0]
    assert len(existing_path.nodes) == 2
    assert existing_path.nodes == [root_node, node1]
    assert node1.path == existing_path
    assert node1.added_to_tree is True
    assert tree.node_count == 2

def test_tree_add_node_fork_path(tree_with_root):
    """Checks adding a node that creates a new path (forks)."""
    tree = tree_with_root
    root_node = tree.paths[0].nodes[0] # Get the root node from the tree

    node1_branch1 = Node(1, 0, 0)
    node1_branch1.parent = root_node
    root_node.children.append(node1_branch1)
    node1_branch1.cost = distance_to(root_node, node1_branch1)
    node1_branch1.log_survival = 0.0
    node1_branch1.p_fail = 0.0
    tree.add_node(node1_branch1, multiple_children=False) # Add first branch

    node1_branch2 = Node(0, 1, 0) # Node at a different location, but parent is root
    node1_branch2.parent = root_node
    root_node.children.append(node1_branch2)
    node1_branch2.cost = distance_to(root_node, node1_branch2)
    node1_branch2.log_survival = 0.0
    node1_branch2.p_fail = 0.0

    # Add node1_branch2, should create a new path because multiple_children is True
    tree.add_node(node1_branch2, multiple_children=True)

    assert len(tree.paths) == 2 # Should have two paths now

    # Check that the paths contain the correct nodes
    path_nodes = [path.nodes for path in tree.paths]
    assert [root_node, node1_branch1] in path_nodes
    assert [root_node, node1_branch2] in path_nodes

    # Check path references for the added nodes
    assert node1_branch1.path in tree.paths
    assert node1_branch2.path in tree.paths
    assert node1_branch1.path != node1_branch2.path # They should be on different paths
    assert node1_branch2.added_to_tree is True
    assert tree.node_count == 3

def test_tree_nearest(tree_with_root):
    """Checks the tree.nearest function."""
    tree = tree_with_root
    root_node = tree.paths[0].nodes[0]

    node1 = Node(1, 1, 0)
    node1.parent = root_node
    root_node.children.append(node1)
    tree.add_node(node1, multiple_children=False)

    node2 = Node(5, 5, 0)
    node2.parent = root_node
    root_node.children.append(node2)
    tree.add_node(node2, multiple_children=True) # Create a fork

    # Test nearest to a point near root
    rand_node_near_root = Node(0.1, 0.1, 0)
    nearest = tree.nearest(rand_node_near_root)
    assert nearest == root_node

    # Test nearest to a point near node1
    rand_node_near_node1 = Node(1.1, 1.1, 0)
    nearest = tree.nearest(rand_node_near_node1)
    assert nearest == node1

    # Test nearest to a point near node2
    rand_node_near_node2 = Node(4.9, 4.9, 0)
    nearest = tree.nearest(rand_node_near_node2)
    assert nearest == node2

def test_tree_neighbors(tree_with_root):
    """Checks the tree.neighbors function."""
    tree = tree_with_root
    root_node = tree.paths[0].nodes[0]

    node1_close = Node(0.1, 0.1, 0)
    node1_close.parent = root_node
    root_node.children.append(node1_close)
    tree.add_node(node1_close, multiple_children=False)

    node2_far = Node(1.0, 1.0, 0)
    node2_far.parent = root_node
    root_node.children.append(node2_far)
    tree.add_node(node2_far, multiple_children=True) # Fork

    node3_close_to_node1 = Node(0.2, 0.1, 0)
    node3_close_to_node1.parent = node1_close
    node1_close.children.append(node3_close_to_node1)
    tree.add_node(node3_close_to_node1, multiple_children=False)

    # PARETO_RADIUS is 0.25 in your code

    # Neighbors of root_node (0,0,0)
    neighbors_of_root = tree.neighbors(root_node)
    # Should include node1_close (distance sqrt(0.1^2 + 0.1^2) approx 0.14 < 0.25)
    # Should include node3_close_to_node1 (distance sqrt(0.2^2 + 0.1^2) approx 0.22 < 0.25)
    # Should NOT include node2_far (distance sqrt(1^2 + 1^2) approx 1.41 > 0.25)
    assert node1_close in neighbors_of_root
    assert node3_close_to_node1 in neighbors_of_root
    assert node2_far not in neighbors_of_root
    assert len(neighbors_of_root) == 2

    # Neighbors of node1_close (0.1, 0.1, 0)
    neighbors_of_node1_close = tree.neighbors(node1_close)
    # Should include root_node (distance approx 0.14 < 0.25)
    # Should include node3_close_to_node1 (distance sqrt(0.1^2 + 0^2) = 0.1 < 0.25)
    assert root_node in neighbors_of_node1_close
    assert node3_close_to_node1 in neighbors_of_node1_close
    assert node2_far not in neighbors_of_node1_close
    assert len(neighbors_of_node1_close) == 3

    # Neighbors of node2_far (1.0, 1.0, 0)
    neighbors_of_node2_far = tree.neighbors(node2_far)
    # Should not include root_node, node1_close, node3_close_to_node1
    assert len(neighbors_of_node2_far) == 0

def test_tree_get_path_to(tree_with_root):
    """Checks the tree.get_path_to function."""
    tree = tree_with_root
    root_node = tree.paths[0].nodes[0]

    node1 = Node(1, 0, 0)
    node1.parent = root_node
    root_node.children.append(node1)
    tree.add_node(node1, multiple_children=False)

    node2 = Node(2, 0, 0)
    node2.parent = node1
    node1.children.append(node2)
    tree.add_node(node2, multiple_children=False)

    node3 = Node(0, 1, 0) # Forking node
    node3.parent = root_node
    root_node.children.append(node3)
    tree.add_node(node3, multiple_children=True)

    # Get path to node2
    path_to_node2 = tree.get_path_to(node2)
    assert path_to_node2 is not None
    assert path_to_node2.nodes == [root_node, node1, node2]

    # Get path to node3
    path_to_node3 = tree.get_path_to(node3)
    assert path_to_node3 is not None
    assert path_to_node3.nodes == [root_node, node3]

    # Get path to a node not in the tree
    non_existent_node = Node(10, 10, 0)
    path_to_non_existent = tree.get_path_to(non_existent_node)
    assert path_to_non_existent is None
