import pytest
import numpy as np
from helper_functions import distance_to, get_coord, is_collision_free, steer
from PO_RRT_Star import Node, Grid, GRID_WIDTH, GRID_HEIGHT, X_MIN, Y_MIN, X_MAX, Y_MAX, GRID_RESOLUTION, DEFAULT_STEP_SIZE

# Create a dummy Grid for collision checks using a fixture
@pytest.fixture
def dummy_grid_instance_with_obstacles():
    """Provides a Grid instance with obstacles for collision checks."""
    dummy_obstacles = [
        {"type": "circular", "center": (1.0, 1.0), "radius": 0.3, "safe_dist": 0.1},
        {"type": "rectangular", "x_range": (2.5, 3.5), "y_range": (2.0, 3.0), "probability": 0.2}
    ]
    return Grid(GRID_WIDTH, GRID_HEIGHT, dummy_obstacles)

def test_distance_to_nodes():
    """Checks the distance_to function with Node objects."""
    node1 = Node(0, 0, 0)
    node2 = Node(3, 4, np.pi/4) # Theta doesn't affect distance
    assert np.isclose(distance_to(node1, node2), 5.0)

    node3 = Node(1, 1, 0)
    node4 = Node(1, 1, np.pi)
    assert np.isclose(distance_to(node3, node4), 0.0)

def test_distance_to_tuples():
    """Checks the distance_to function with tuple coordinates."""
    coord1 = (0, 0)
    coord2 = (3, 4)
    assert np.isclose(distance_to(coord1, coord2), 5.0)

    coord3 = (1, 1)
    coord4 = (1, 1)
    assert np.isclose(distance_to(coord3, coord4), 0.0)

def test_get_coord():
    """Checks the get_coord function."""
    node = Node(5, 10, np.pi/2)
    assert get_coord(node) == (5, 10) # get_coord in helper_functions only returns x, y

def test_is_collision_free_free_space(dummy_grid_instance_with_obstacles):
    """Checks is_collision_free for a node in free space."""
    free_node = Node(0.1, 0.1, 0)
    assert is_collision_free(free_node, dummy_grid_instance_with_obstacles) == True

def test_is_collision_free_in_circular_obstacle(dummy_grid_instance_with_obstacles):
    """Checks is_collision_free for a node inside a circular obstacle (high probability)."""
    obstacle_node_circ = Node(1.0, 1.0, 0)
    # is_collision_free checks if grid value < 0.3
    assert is_collision_free(obstacle_node_circ, dummy_grid_instance_with_obstacles) == False

def test_is_collision_free_in_rectangular_obstacle(dummy_grid_instance_with_obstacles):
    """Checks is_collision_free for a node inside a rectangular obstacle (probability > 0.0)."""
    obstacle_node_rect = Node(3.0, 2.5, 0)
    assert is_collision_free(obstacle_node_rect, dummy_grid_instance_with_obstacles) == True

def test_is_collision_free_in_safe_distance_zone(dummy_grid_instance_with_obstacles):
    """Checks is_collision_free for a node in the safe distance zone (probability > 0.0)."""
    safe_dist_node = Node(1.35, 1.0, 0) # Just outside radius 0.3 + safe_dist 0.1 = 0.4. 1.0 + 0.35 = 1.35
    assert is_collision_free(safe_dist_node, dummy_grid_instance_with_obstacles) == True
    # The grid value at this point should be < 0.3, so it should be free

def test_is_collision_free_outside_grid(dummy_grid_instance_with_obstacles):
    """Checks is_collision_free for a node outside the grid boundaries."""
    outside_node_x = Node(X_MAX + 1.0, Y_MIN, 0)
    outside_node_y = Node(X_MIN, Y_MIN - 1.0, 0)
    assert is_collision_free(outside_node_x, dummy_grid_instance_with_obstacles) == False
    assert is_collision_free(outside_node_y, dummy_grid_instance_with_obstacles) == False


def test_steer_within_step_size():
    """Checks steer when distance is less than step size."""
    from_node = Node(0, 0, 0)
    to_node = Node(0.5, 0, 0)
    step_size = 1.0
    x, y, theta = steer(from_node, to_node, step_size)
    # Will not reach to_node, but will be at step_size distance from from_node
    assert np.isclose(x, 1)
    assert np.isclose(y, 0.0)
    assert np.isclose(theta, 0.0) # Theta from from_node

def test_steer_at_step_size():
    """Checks steer when distance is exactly equal to step size."""
    from_node = Node(0, 0, 0)
    to_node = Node(1.0, 0, 0)
    step_size = 1.0
    x, y, theta = steer(from_node, to_node, step_size)
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 0.0)
    assert np.isclose(theta, 0.0)

def test_steer_beyond_step_size():
    """Checks steer when distance is greater than step size."""
    from_node = Node(0, 0, 0)
    to_node = Node(2, 0, 0)
    step_size = 1.0
    x, y, theta = steer(from_node, to_node, step_size)
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 0.0)
    assert np.isclose(theta, 0.0) # Theta from to_node

def test_steer_with_angle():
    """Checks steer with angled movement."""
    from_node_angle = Node(0, 0, np.pi/4)
    to_node_angle = Node(2, 2, np.pi/2)
    step_size = np.sqrt(2) # Distance to (1,1) from (0,0)
    x, y, theta = steer(from_node_angle, to_node_angle, step_size)
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 1.0)
    assert np.isclose(theta, np.pi/4) # steer in helper_functions calculates new theta based on dx, dy

