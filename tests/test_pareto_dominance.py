import pytest
from PO_RRT_Star import Tree, Grid, GRID_WIDTH, GRID_HEIGHT

# Create a dummy Grid for Tree initialization
@pytest.fixture
def dummy_grid_instance():
    """Provides a basic empty Grid instance for tests."""
    return Grid(GRID_WIDTH, GRID_HEIGHT, [])

# Create a Tree instance
@pytest.fixture
def tree_instance(dummy_grid_instance):
    """Provides a Tree instance for testing Pareto dominance."""
    return Tree(dummy_grid_instance)

def test_pareto_dominance_strict(tree_instance):
    """Checks strict Pareto dominance (both metrics better)."""
    # cost1 < cost2 and fail1 < fail2
    assert tree_instance.pareto_dominates(1.0, 0.1, 2.0, 0.2) is True

def test_pareto_dominance_cost_equal_fail_better(tree_instance):
    """Checks Pareto dominance with equal cost and better failure probability."""
    # cost1 <= cost2 and fail1 < fail2
    assert tree_instance.pareto_dominates(1.0, 0.1, 1.0, 0.2) is True

def test_pareto_dominance_cost_better_fail_equal(tree_instance):
    """Checks Pareto dominance with better cost and equal failure probability."""
    # cost1 < cost2 and fail1 <= fail2
    assert tree_instance.pareto_dominates(1.0, 0.1, 2.0, 0.1) is True

def test_no_dominance_equal(tree_instance):
    """Checks when metrics are equal (no dominance)."""
    assert tree_instance.pareto_dominates(1.0, 0.1, 1.0, 0.1) is False

def test_no_dominance_worse_cost_better_fail(tree_instance):
    """Checks when cost is worse but failure is better (no dominance)."""
    assert tree_instance.pareto_dominates(2.0, 0.1, 1.0, 0.2) is False

def test_no_dominance_better_cost_worse_fail(tree_instance):
    """Checks when cost is better but failure is worse (no dominance)."""
    assert tree_instance.pareto_dominates(1.0, 0.2, 2.0, 0.1) is False

def test_no_dominance_worse_both(tree_instance):
    """Checks when both metrics are worse (no dominance)."""
    assert tree_instance.pareto_dominates(2.0, 0.2, 1.0, 0.1) is False

def test_pareto_dominance_boundary_cases(tree_instance):
    """Checks boundary cases with small differences."""
    assert tree_instance.pareto_dominates(1.0 - 1e-9, 0.1, 1.0, 0.1) is True
    assert tree_instance.pareto_dominates(1.0, 0.1 - 1e-9, 1.0, 0.1) is True
    assert tree_instance.pareto_dominates(1.0 - 1e-9, 0.1 - 1e-9, 1.0, 0.1) is True
    assert tree_instance.pareto_dominates(1.0, 0.1, 1.0 - 1e-9, 0.1) is False # Check dominance in the other direction
