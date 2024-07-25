from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class ScoreNode:
    anls_score: Optional[float] = None
    children: dict[str, Any] = field(default_factory=dict)


def construct_nested_dict(
    list_of_dicts: list[dict[tuple[str, ...], float]]
) -> dict[str, ScoreNode]:
    """Construct a nested dictionary from a list of dictionaries with nested keys.

    Note: If there are duplicates of keys in the list of dictionaries, the last value will be used.

    Args:
        list_of_dicts: A list of dictionaries with nested keys.

    Returns:
        A nested dictionary.

    Example:
        >>> list_of_dicts = [
                {("a",): 3},
                {("a", "b", "c"): 1},
                {("a", "b", "d"): 2},
                {("a", "c", "e"): 3},
            ],
        >>> construct_nested_dict(list_of_dicts)
            {
                "a": ScoreNode(
                    anls_score=3,
                    children={
                        "b": ScoreNode(
                            children={
                                "c": ScoreNode(anls_score=1),
                                "d": ScoreNode(anls_score=2),
                            }
                        ),
                        "c": ScoreNode(children={"e": ksu.ScoreNode(anls_score=3)}),
                    },
                )
            },
    """
    nested_dict: dict[str, ScoreNode] = {}

    if len(list_of_dicts) == 0:
        return nested_dict

    for entry in list_of_dicts:
        for key_tuple, value in entry.items():
            current_dict: dict[str, ScoreNode] = nested_dict
            # Traverse and build nested dict, except for last entry
            for key in key_tuple[:-1]:
                if key not in current_dict:
                    current_dict[key] = ScoreNode()
                current_dict = current_dict[key].children

            # Set the value for the final key
            final_key = key_tuple[-1]
            if final_key not in current_dict:
                current_dict[final_key] = ScoreNode()
            current_dict[final_key].anls_score = value

    return nested_dict


def merge_and_calculate_mean(
    list_of_dicts: list[dict[tuple[str, ...], float]]
) -> list[dict[tuple[str, ...], float]]:
    """
    Merges a list of dictionaries and calculates the mean of values for each key.

    The function takes a list of dictionaries where keys are tuples of strings and
    values are floats. It combines the dictionaries into a single dictionary by
    summing values for the same keys and then calculates the mean value for each key.

    Args:
        list_of_dicts (list): A list of dictionaries with tuple keys and float values.

    Returns:
        list: A list of dictionaries, each containing a single key-value pair where
              values are the mean of the original values for the corresponding key.

    Example:
        >>> list_of_dicts = [
        ...     {('a', 'b'): 10.0, ('c', 'd'): 20.0},
        ...     {('a', 'b'): 30.0, ('e', 'f'): 40.0}
        ... ]
        >>> merge_and_calculate_mean(list_of_dicts)
        [{('a', 'b'): 20.0}, {('c', 'd'): 20.0}, {('e', 'f'): 40.0}]
    """
    combined_scores: dict[tuple[str, ...], float] = {}
    count_dict: dict[tuple[str, ...], int] = {}

    for d in list_of_dicts:
        for k, v in d.items():
            if k not in combined_scores:
                combined_scores[k] = 0
                count_dict[k] = 0
            combined_scores[k] += v
            count_dict[k] += 1

    for k in combined_scores.keys():
        combined_scores[k] /= count_dict[k]

    # Converting the combined_scores back to a list of dictionaries
    list_combined_scores = [{k: v} for k, v in combined_scores.items()]

    return list_combined_scores
