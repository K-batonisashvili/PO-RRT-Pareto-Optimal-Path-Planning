import pytest
import numpy as np
from PO_RRT_star_occupancy import Grid, GRID_WIDTH, GRID_HEIGHT, X_MIN, Y_MIN, GRID_RESOLUTION

# Define a fixture for a grid with specific obstacles
@pytest.fixture
def grid_with_obstacles():
    """Provides a Grid instance with predefined obstacles."""
    obstacles = [
        {"type": "circular", "center": (1.0, 1.0), "radius": 0.5, "safe_dist": 0.2},
        {"type": "rectangular", "x_range": (2.5, 3.5), "y_range": (2.0, 3.0), "probability": 0.2}
    ]
    return Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

def test_add_circular_obstacle(grid_with_obstacles):
    """Checks if a circular obstacle is added correctly to the grid."""
    grid = grid_with_obstacles

    # Check a point inside the obstacle (should have high probability)
    # Convert world coordinates to grid indices
    ox, oy = 1.0, 1.0
    center_x_idx = int((ox - X_MIN) / GRID_RESOLUTION)
    center_y_idx = int((oy - Y_MIN) / GRID_RESOLUTION)

    assert grid.grid[center_x_idx][center_y_idx] >= 0.9

    # Point just outside the radius but within safe_dist (should have probability > 0 and < 0.9)
    # The probability calculation is linear decay in the safe_dist zone
    test_x = ox + 0.55 # radius + a bit into safe_dist (0.5 + 0.05)
    test_y = oy
    test_x_idx = int((test_x - X_MIN) / GRID_RESOLUTION)
    test_y_idx = int((test_y - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[test_x_idx][test_y_idx] > 0.0
    assert grid.grid[test_x_idx][test_y_idx] < 0.9

    # Point far outside the safe_dist (should have probability 0)
    test_x_far = ox + 1.0 # well outside radius + safe_dist (0.5 + 0.2 = 0.7)
    test_y_far = oy
    test_x_far_idx = int((test_x_far - X_MIN) / GRID_RESOLUTION)
    test_y_far_idx = int((test_y_far - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[test_x_far_idx][test_y_far_idx] == 0.0

def test_add_unknown_area(grid_with_obstacles):
    """Checks if a rectangular unknown area is added correctly to the grid."""
    grid = grid_with_obstacles

    # Check a point inside the rectangular area
    test_x, test_y = 3.0, 2.5 # Point within (2.5, 3.5) and (2.0, 3.0)
    test_x_idx = int((test_x - X_MIN) / GRID_RESOLUTION)
    test_y_idx = int((test_y - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[test_x_idx][test_y_idx] == 0.2

    # Check a point outside the rectangular area
    test_x_out, test_y_out = 0.5, 0.5
    test_x_out_idx = int((test_x_out - X_MIN) / GRID_RESOLUTION)
    test_y_out_idx = int((test_y_out - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[test_x_out_idx][test_y_out_idx] == 0.0

def test_multiple_obstacles_interaction():
    """Checks if multiple obstacles are added and probabilities are handled (highest takes precedence)."""
    # Create obstacles that overlap or are close
    obstacles = [
        {"type": "circular", "center": (1.0, 1.0), "radius": 0.3, "safe_dist": 0.1}, # High risk near center
        {"type": "rectangular", "x_range": (0.8, 1.2), "y_range": (0.8, 1.2), "probability": 0.5} # Lower probability rectangular area overlapping the circle
    ]
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    # Check a point at the center of the circle, should have high probability from circle
    center_x_idx = int((1.0 - X_MIN) / GRID_RESOLUTION)
    center_y_idx = int((1.0 - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[center_x_idx][center_y_idx] >= 0.9

    # Check a point in the rectangular area but outside the circular obstacle's safe_dist
    test_x, test_y = 1.1, 1.1 # Within rectangular area, outside circle radius+safe_dist
    test_x_idx = int((test_x - X_MIN) / GRID_RESOLUTION)
    test_y_idx = int((test_y - Y_MIN) / GRID_RESOLUTION)
    # This point is within the rectangular area, should have its probability
    assert grid.grid[test_x_idx][test_y_idx] >= 0.5

    # Check a point in free space
    free_x_idx = int((0.1 - X_MIN) / GRID_RESOLUTION)
    free_y_idx = int((0.1 - Y_MIN) / GRID_RESOLUTION)
    assert grid.grid[free_x_idx][free_y_idx] == 0.0