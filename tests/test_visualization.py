import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from visualization import init_progress_plot_3d, update_progress_plot_3d, plot_paths_metrics
from PO_RRT_Star import Node, GRID_WIDTH, GRID_HEIGHT, X_MIN, X_MAX, Y_MIN, Y_MAX

# Mock the matplotlib.pyplot module as visualization functions interact with it
@pytest.fixture(autouse=True)
def mock_matplotlib(monkeypatch):
    """Mocks matplotlib.pyplot to prevent actual plotting during tests."""
    mock_plt = MagicMock()
    # Mock necessary methods that visualization functions call
    mock_plt.figure.return_value = MagicMock()
    mock_plt.figure.return_value.add_subplot.return_value = MagicMock() # Mock the axes object
    mock_plt.show.return_value = None
    mock_plt.pause.return_value = None
    mock_plt.draw.return_value = None
    mock_plt.subplots.return_value = (MagicMock(), MagicMock()) # For plot_paths_metrics

    monkeypatch.setattr('visualization.plt', mock_plt)
    return mock_plt

def test_init_progress_plot_3d(mock_matplotlib):
    """Checks if init_progress_plot_3d calls expected matplotlib functions."""
    start = (0.5, 0.5, 0)
    goal = (3.5, 3.5, 0)
    x_lim = (X_MIN, X_MAX)
    y_lim = (Y_MIN, Y_MAX)
    obstacles = [
        {"type": "circular", "center": (2.0, 2.0), "radius": 0.5, "safe_dist": 0.2},
        {"type": "rectangular", "x_range": (1.0, 1.5), "y_range": (1.0, 1.5), "probability": 0.1}
    ]
    z_lim = (0.0, 1.0)

    fig, ax, lc, edge_segments = init_progress_plot_3d(start, goal, x_lim, y_lim, obstacles, z_lim)

    # Check if figure and axes were created
    mock_matplotlib.figure.assert_called_once()
    mock_matplotlib.figure.return_value.add_subplot.assert_called_once_with(projection='3d')

    # Check if axis limits were set
    ax.set_xlim.assert_called_once_with(*x_lim)
    ax.set_ylim.assert_called_once_with(*y_lim)
    ax.set_zlim.assert_called_once_with(*z_lim)

    # Check if start and goal points were scattered
    ax.scatter.assert_any_call(start[0], start[1], 0.0, c='green', s=50, label='Start')
    ax.scatter.assert_any_call(goal[0], goal[1], 0.0, c='blue', s=50, label='Goal')

    # Check if obstacles were plotted (basic check for method calls)
    ax.plot.assert_called() # For circular obstacle boundary
    ax.plot_trisurf.assert_called() # For rectangular obstacle area

    # Check if legend was called
    ax.legend.assert_called_once()

    # Check if Line3DCollection was created and added
    # We can't directly check the Line3DCollection constructor call easily without more complex mocking
    # But we can check if ax.add_collection was called
    ax.add_collection.assert_called_once()

    # Check if show and ion were called
    mock_matplotlib.ion.assert_called_once()
    mock_matplotlib.show.assert_called_once()

    # Check the return types (mock objects)
    assert isinstance(fig, MagicMock)
    assert isinstance(ax, MagicMock)
    # lc and edge_segments are created within the function, check their types
    # lc is a Line3DCollection mock, edge_segments is a list
    # We can't easily check the type of lc unless we mock Line3DCollection itself
    # But we can check that edge_segments is a list
    assert isinstance(edge_segments, list)


def test_update_progress_plot_3d(mock_matplotlib):
    """Checks if update_progress_plot_3d appends segment and updates plot."""
    # Create dummy Line3DCollection and edge_segments list
    mock_lc = MagicMock()
    edge_segments = []
    parent_node = Node(0, 0, 0)
    parent_node.p_fail = 0.0
    new_node = Node(1, 1, 0)
    new_node.p_fail = 0.1

    update_progress_plot_3d(mock_lc, edge_segments, parent_node, new_node, pause_time=0.01)

    # Check if the new segment was added to the list
    expected_segment = [(parent_node.x, parent_node.y, parent_node.p_fail), (new_node.x, new_node.y, new_node.p_fail)]
    assert edge_segments == [expected_segment]

    # Check if set_segments was called on the Line3DCollection with the updated list
    mock_lc.set_segments.assert_called_once_with(edge_segments)

    # Check if plot update functions were called
    mock_matplotlib.draw.assert_called_once()
    mock_matplotlib.pause.assert_called_once_with(0.01)


def test_plot_paths_metrics(mock_matplotlib):
    """Checks if plot_paths_metrics extracts data and calls expected matplotlib functions."""
    # Create a list of dummy paths (as returned by rrt_star)
    paths = [
        ([Node(0,0,0), Node(1,0,0)], 1.0, 0.1),
        ([Node(0,0,0), Node(0,1,0), Node(1,1,0)], 2.0, 0.05),
    ]

    plot_paths_metrics(paths)

    # Check if subplots was called
    mock_matplotlib.subplots.assert_called_once_with(figsize=(8, 6))

    # Get the mock axis object from the subplots return value
    mock_ax = mock_matplotlib.subplots.return_value[1]

    # Check if ax.clear was called
    mock_ax.clear.assert_called_once()

    # Check if scatter was called with the correct data
    expected_costs = [1.0, 2.0]
    expected_pfails = [0.1, 0.05]
    mock_ax.scatter.assert_called_once_with(expected_costs, expected_pfails, marker='o', color='blue', label='Paths')

    # Check if labels and title were set
    mock_ax.set_xlabel.assert_called_once_with('Total Cost')
    mock_ax.set_ylabel.assert_called_once_with('Failure Probability')
    mock_ax.set_title.assert_called_once_with('Cost vs. Failure Probability for Extracted Paths')

    # Check if grid and legend were called
    mock_ax.grid.assert_called_once_with(True)
    mock_ax.legend.assert_called_once()

    # Check if show was called with block=True
    mock_matplotlib.show.assert_called_once_with(block=True)

