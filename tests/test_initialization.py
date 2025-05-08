import pytest
import numpy as np
from PO_RRT_Star import Node, Path, Tree, Grid, GRID_WIDTH, GRID_HEIGHT, X_MIN, Y_MIN, X_MAX, Y_MAX, GRID_RESOLUTION

# Define a fixture for a dummy grid, reusable across tests
@pytest.fixture
def dummy_grid_instance():
    """Provides a basic empty Grid instance for tests."""
    obstacles = [] # Initialize with no obstacles
    return Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

def test_node_initialization():
    """Checks if a Node is initialized correctly."""
    node = Node(1.0, 2.0, np.pi/2)
    assert node.x == 1.0
    assert node.y == 2.0
    assert node.theta == np.pi/2
    assert node.parent is None
    assert len(node.children) == 0
    assert node.cost == 0.0
    assert node.p_fail == 0.0
    assert node.log_survival == 0.0
    assert node.added_to_tree is False
    assert node.path is None
    assert node.is_goal is False

def test_path_initialization():
    """Checks if a Path is initialized correctly."""
    path = Path()
    assert len(path.nodes) == 0
    assert path.cost == 0.0 # Cost of empty path is 0
    assert path.p_fail == 1.0 # Failure probability of empty path is also 0 because no nodes are present

def test_tree_initialization(dummy_grid_instance):
    """Checks if a Tree is initialized correctly."""
    tree = Tree(dummy_grid_instance)
    assert len(tree.paths) == 0
    assert tree.rewire_counts == 0
    assert tree.rewire_neighbors_count == 0
    assert tree.node_count == 0

def test_grid_initialization():
    """Checks if a Grid is initialized correctly with dimensions and empty grid."""
    obstacles = [] # Initialize with no obstacles
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)
    assert grid.width == GRID_WIDTH
    assert grid.height == GRID_HEIGHT
    assert np.sum(grid.grid) == 0.0 # Check if grid is initially all zeros
    assert grid.resolution == GRID_RESOLUTION
    assert grid.x_min == X_MIN
    assert grid.x_max == X_MAX
    assert grid.y_min == Y_MIN
    assert grid.y_max == Y_MAX
    assert grid.obstacles == obstacles

def test_grid_dimensions_match_constants():
    """Checks that the calculated grid dimensions match the constants."""
    assert GRID_WIDTH == int((X_MAX - X_MIN) / GRID_RESOLUTION)
    assert GRID_HEIGHT == int((Y_MAX - Y_MIN) / GRID_RESOLUTION)

