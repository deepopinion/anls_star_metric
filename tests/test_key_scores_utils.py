import sys

sys.path.append("../src")

import pytest

import anls_star.key_scores_utils as ksu


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
    assert all(isinstance(item, dict) for item in result), "All items in result should be dictionaries"
    assert all(len(item) == 1 for item in result), "Each dictionary in result should have exactly one key-value pair"
    assert all(isinstance(list(item.keys())[0], tuple) for item in result), "All keys should be tuples"
    assert all(isinstance(list(item.values())[0], float) for item in result), "All values should be floats"


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([], {}),
        (
            [
                {("a",): 3},
                {("a", "b", "c"): 1},
                {("a", "b", "d"): 2},
                {("a", "c", "e"): 3},
            ],
            {
                "a": ksu.ScoreNode(
                    anls_score=3,
                    children={
                        "b": ksu.ScoreNode(
                            children={
                                "c": ksu.ScoreNode(anls_score=1),
                                "d": ksu.ScoreNode(anls_score=2),
                            }
                        ),
                        "c": ksu.ScoreNode(children={"e": ksu.ScoreNode(anls_score=3)}),
                    },
                )
            },
        ),
        (
            [{("x",): 5.0}, {("y",): 10.0}],
            {"x": ksu.ScoreNode(anls_score=5.0), "y": ksu.ScoreNode(anls_score=10.0)},
        ),
        (
            [{("a", "b"): 1}, {("a", "b"): 2}],
            {"a": ksu.ScoreNode(children={"b": ksu.ScoreNode(anls_score=2)})},
        ),
    ],
)
def test_construct_nested_dict(
    input_list: list[dict[tuple[str, ...], float]],
    expected_output: dict,
):
    result = ksu.construct_nested_dict(input_list)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
