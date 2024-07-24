import sys

sys.path.append("../src")

import src.key_scores_utils as ksu
import pytest


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([], []),
        (
            [
                {("a", "b"): 10.0, ("c", "d"): 20.0},
                {("a", "b"): 30.0, ("e", "f"): 40.0},
            ],
            [{("a", "b"): 20.0}, {("c", "d"): 20.0}, {("e", "f"): 40.0}],
        ),
        (
            [{("x", "y"): 5.0}, {("x", "y"): 15.0}, {("x", "y"): 25.0}],
            [{("x", "y"): 15.0}],
        ),
        (
            [{("a",): 1.0, ("b",): 2.0}, {("a",): 3.0, ("c",): 4.0}],
            [{("a",): 2.0}, {("b",): 2.0}, {("c",): 4.0}],
        ),
    ],
)
def test_merge_and_calculate_mean(
    input_list: list[dict[tuple[str, ...], float]],
    expected_output: list[dict[tuple[str, ...], float]],
):
    result = ksu.merge_and_calculate_mean(input_list)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    assert isinstance(result, list), "Result should be a list"
    assert all(
        isinstance(item, dict) for item in result
    ), "All items in result should be dictionaries"
    assert all(
        len(item) == 1 for item in result
    ), "Each dictionary in result should have exactly one key-value pair"
    assert all(
        isinstance(list(item.keys())[0], tuple) for item in result
    ), "All keys should be tuples"
    assert all(
        isinstance(list(item.values())[0], float) for item in result
    ), "All values should be floats"
