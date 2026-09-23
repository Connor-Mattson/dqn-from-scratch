"""Checks for the across-seed statistics (provided plumbing).

The t-distribution tail is implemented in-repo rather than taken from scipy, so it is
pinned here against published t-table values.
"""

import numpy as np
import pytest

from dqn.stats import aggregate_seeds, mean_stderr, t_two_sided_p, welch_ttest

pytestmark = pytest.mark.scaffold


@pytest.mark.parametrize(
    "t, df, expected",
    [
        (2.228139, 10, 0.05),  # two-sided 5% critical value, df=10
        (3.182446, 3, 0.05),
        (2.776445, 4, 0.05),
        (2.570582, 5, 0.05),
        (4.604095, 4, 0.01),
        (1.812461, 10, 0.10),
        (2.0, 10, 0.073388),
        (1.0, 1, 0.5),
        (0.0, 5, 1.0),
        (1.959964, 1e7, 0.05),  # converges on the normal quantile for large df
    ],
)
def test_t_tail_matches_published_critical_values(t, df, expected):
    assert t_two_sided_p(t, df) == pytest.approx(expected, abs=2e-5)


def test_t_tail_is_symmetric_and_decreasing():
    assert t_two_sided_p(-2.0, 8) == pytest.approx(t_two_sided_p(2.0, 8))
    assert t_two_sided_p(3.0, 8) < t_two_sided_p(2.0, 8) < t_two_sided_p(1.0, 8)


def test_welch_ttest_on_a_hand_checkable_sample():
    # Both arms have variance 2.5, n=5, so se^2 = 0.5 each and df collapses to 8 exactly.
    result = welch_ttest([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])

    assert result is not None
    assert result.difference == pytest.approx(5.0)
    assert result.stderr == pytest.approx(1.0)
    assert result.t == pytest.approx(5.0)
    assert result.df == pytest.approx(8.0)
    assert result.significant
    assert 0.0005 < result.p < 0.002


def test_welch_ttest_finds_no_difference_between_overlapping_arms():
    result = welch_ttest([10, 12, 11, 13, 9], [11, 10, 12, 12, 10])

    assert result is not None
    assert not result.significant
    assert result.p > 0.3


def test_welch_ttest_refuses_a_single_seed():
    # One run per arm has no within-arm variance; any p-value would be invented.
    assert welch_ttest([1.0], [5.0]) is None
    assert welch_ttest([1.0, 2.0], [5.0]) is None


def test_welch_ttest_handles_two_constant_arms():
    result = welch_ttest([3.0, 3.0], [3.0, 3.0])

    assert result is not None
    assert result.difference == 0.0
    assert not result.significant  # p is nan, which is not < 0.05


def test_mean_stderr():
    mean, stderr = mean_stderr([2.0, 4.0, 6.0])
    assert mean == pytest.approx(4.0)
    assert stderr == pytest.approx(np.std([2.0, 4.0, 6.0], ddof=1) / np.sqrt(3))
    assert mean_stderr([7.0]) == (7.0, 0.0)


def test_aggregate_seeds_averages_onto_a_shared_grid():
    # Two seeds on different x-axes: a flat 10 and a flat 20 average to 15 everywhere.
    band = aggregate_seeds([(np.array([0, 50, 100]), np.array([10.0, 10.0, 10.0])),
                            (np.array([0, 30, 90]), np.array([20.0, 20.0, 20.0]))])

    assert band is not None
    assert band.n_seeds == 2
    assert band.has_band
    np.testing.assert_allclose(band.mean, 15.0)
    # se of {10, 20} is 5.0, constant across the grid.
    np.testing.assert_allclose(band.stderr, 5.0)


def test_aggregate_seeds_spans_only_the_range_every_seed_covers():
    band = aggregate_seeds([(np.array([10, 100]), np.array([1.0, 2.0])),
                            (np.array([40, 80]), np.array([1.0, 2.0]))])

    assert band is not None
    # Latest start and earliest end, so the seed count behind the mean never changes.
    assert band.x[0] == pytest.approx(40.0)
    assert band.x[-1] == pytest.approx(80.0)


def test_aggregate_seeds_of_one_run_has_no_band():
    band = aggregate_seeds([(np.array([0, 10]), np.array([1.0, 3.0]))])

    assert band is not None
    assert band.n_seeds == 1
    assert not band.has_band
    np.testing.assert_allclose(band.stderr, 0.0)


def test_aggregate_seeds_rejects_unusable_input():
    assert aggregate_seeds([]) is None
    assert aggregate_seeds([(np.array([5]), np.array([1.0]))]) is None  # single point
    assert aggregate_seeds([(np.array([7, 7]), np.array([1.0, 2.0]))]) is None  # zero width
