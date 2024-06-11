import itertools
import random
import sys

sys.path.append("../src")
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from src.anls_star import ANLSTree, anls_score


#### Helper functions ####
def has_tuple(obj):
    if isinstance(obj, tuple):
        return True
    if isinstance(obj, dict):
        return any(has_tuple(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_tuple(v) for v in obj)
    return False


#### Tests ####
### Scalars ###
@given(st.none() | st.booleans() | st.text() | st.floats() | st.integers())
def test_anls_same_value(s):
    assert anls_score(s, s, return_gt=True) == (approx(1.0), s)


@given(st.booleans() | st.text() | st.floats() | st.integers())
def test_anls_cast_to_str(s):
    assert anls_score(s, str(s), return_gt=True) == (approx(1.0), s)
    assert anls_score(str(s), s, return_gt=True) == (approx(1.0), str(s))


def test_anls_score_should_trim_whitespace():
    anls, closest_gt = anls_score("Test Hello ", " Test    Hello\n\n", return_gt=True)
    assert anls == approx(1.0)
    assert closest_gt == "Test Hello "


def test_anls_score_case_insensitive():
    anls, closest_gt = anls_score("TeSt HeLlO", "tEsT hElLO", return_gt=True)
    assert anls == approx(1.0)
    assert closest_gt == "TeSt HeLlO"


### Multiple GTs (tuple) ###
@given(st.none() | st.booleans() | st.text() | st.floats() | st.integers())
def test_anls_score_single_answer(s):
    gts = (s,)
    pred = s
    anls, closest_gt = anls_score(gts, pred, return_gt=True)

    assert anls == approx(1.0)
    # Equal unless both are NaN (NaN != NaN)
    assert closest_gt == s or (closest_gt != closest_gt and s != s)


@given(st.text().filter(lambda x: x != "Hi there"))
def test_anls_score_single_wrong_answer(s):
    gts = (s,)
    answer = "Hi there"
    anls, closest_gt = anls_score(gts, answer, return_gt=True)

    assert anls == approx(0.0)
    assert closest_gt == s


def test_anls_score_single_answer_slightly_off():
    answers = ("500",)
    answer = "$500"
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls < 1.0
    assert anls > 0.3
    assert closest_gt == "500"


def test_anls_score_float_str():
    answers = ("0.123",)
    answer = 0.123
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == "0.123"


@given(st.tuples(st.text()), st.text())
def test_anls_score_single_answer_multiple_possibilities_should_succed(others, correct):
    gts = others + (correct,)
    pred = correct
    anls, closest_gt = anls_score(gts, pred, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == correct


def test_anls_tuple_with_empty_list():
    gt = {"a": ([], ["a", "b", "c"], ["d", "e", "f"])}
    pred = {"a": []}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == {"a": []}


def test_anls_tuple_with_empty_list_2():
    # Same as above, but with the empty dict in the back.
    # This changes the result of the max() call if all options have the same ANLS sum (0.0)
    gt = {"a": (["a", "b", "c"], ["d", "e", "f"], [])}
    pred = {"a": []}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == {"a": []}


def test_anls_empty_tuple_should_fail():
    with pytest.raises(
        ValueError, match="Expected at least 1 valid ground truth option"
    ):
        ANLSTree.make_tree(tuple(), is_gt=True)


### Lists ###
def test_anls_score_empty_list():
    assert anls_score([], [], return_gt=True) == (approx(1.0), [])


@given(st.lists(st.text(), max_size=10))
def test_anls_score_permuted_list(lst: list):
    gt = lst.copy()
    pred = lst.copy()
    random.shuffle(pred)
    assert anls_score(gt, pred, return_gt=True) == (approx(1.0), pred)


@given(st.lists(st.text(), min_size=10, max_size=10))
def test_anls_score_list_results_one_missing_at_end(lst: list):
    assert anls_score(lst, lst[:-1], return_gt=True) == (approx(0.9), lst)


@given(st.lists(st.text(), min_size=10, max_size=10))
def test_anls_score_list_results_one_missing_at_beginning(lst: list):
    """If an element is missing at the beginning, we expect it to appear at
    the end in the best-matching gt since non-matched items are moved back."""
    assert anls_score(lst, lst[1:], return_gt=True) == (approx(0.9), lst[1:] + lst[:1])


def test_anls_score_list_results_completely_off():
    answers = ["a", "b", "c"]
    answer = ["what", "is", "deepopinion"]
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(0.0)
    assert closest_gt == ["a", "b", "c"]


def test_anls_missed_list_elements_should_lower_score():
    gt = ["one", "two", "three"]
    pred1 = ["one", "to", "three"]
    pred2 = ["one", "to"]

    anls1, closest_gt_1 = anls_score(gt, pred1, return_gt=True)
    anls2, closest_gt_2 = anls_score(gt, pred2, return_gt=True)

    assert 1.0 > anls1 > anls2 > 0.0
    assert closest_gt_1 == ["one", "two", "three"]
    assert closest_gt_2 == ["one", "two", "three"]


def test_anls_hallucinated_list_elements_should_lower_score():
    gt = ["one", "two", "three"]
    pred1 = ["one", "to", "three"]
    pred2 = ["one", "to", "three", "four"]

    anls1, closest_gt_1 = anls_score(gt, pred1, return_gt=True)
    anls2, closest_gt_2 = anls_score(gt, pred2, return_gt=True)

    assert 1.0 > anls1 > anls2 > 0.0
    assert closest_gt_1 == gt
    assert closest_gt_2 == gt


def test_anls_list_permutation_invariant():
    gt = ["one", "two", "three"]
    pred = ["three", "one", "two"]
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == pred


### None's ###
@given(st.sampled_from([None, [], {}, ""]))
def test_anls_score_falsy_values_should_match_none(pred):
    """If the pred is "None-y", it should match and be returned as the best match."""
    assert anls_score(None, pred, return_gt=True) == (approx(1.0), pred)


@given(
    st.booleans()
    | st.text(min_size=1)
    | st.floats()
    | st.integers()
    | st.lists(st.text(), min_size=1)
    | st.dictionaries(st.text(), st.text(), min_size=1)
)
def test_anls_score_none_expected_but_not_predicted(pred):
    """If the pred is not "None-y", it should not match and None is the best match."""
    assert anls_score(None, pred, return_gt=True) == (approx(0.0), None)


### Dicts ###
def test_anls_score_empty_dict():
    assert anls_score(dict(), dict(), return_gt=True) == (approx(1.0), dict())


@given(
    st.dictionaries(
        st.text(),
        st.none() | st.booleans() | st.text(min_size=1) | st.floats() | st.integers(),
    )
)
def test_anls_score_two_equal_dicts(d: dict):
    anls, closest_gt = anls_score(d, d, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == d
    assert list(closest_gt.keys()) == list(d.keys())  # same order


@given(
    st.dictionaries(st.text(), st.text(min_size=1), min_size=1),
)
def test_anls_score_two_dicts_hallucinated_key(d: dict):
    gt = d.copy()
    pred = d.copy()
    gt.pop(list(gt.keys())[0])  # remove a random key from gt
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls < 1.0
    assert anls == approx(len(gt) / len(pred))
    for k, kg in itertools.zip_longest(gt, closest_gt):
        if k is not None and kg is not None:
            assert k == kg
            assert gt[k] == closest_gt[kg]
        else:
            # If we have a missing key, the value should be None-y (which is equivalent to missing in ANLS*)
            assert closest_gt[kg] in (None, [], {}, "")


@given(
    st.dictionaries(st.text(), st.text(min_size=1), min_size=1),
)
def test_anls_score_two_dicts_missing_key(d: dict):
    gt = d.copy()
    pred = d.copy()
    pred.pop(list(pred.keys())[0])  # remove a random key from pred
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls < 1.0
    assert anls == approx(len(pred) / len(gt))
    assert closest_gt == gt
    assert list(closest_gt.keys()) == list(gt.keys())


def test_anls_score_two_dicts_one_none():
    answers = {"a": "b"}
    answer = None
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls < 1.0
    assert closest_gt == answers


@given(
    st.dictionaries(
        st.text(),
        st.dictionaries(st.text(), st.text(min_size=1), min_size=1),
        min_size=1,
    ),
)
def test_anls_score_recursive_dicts(d: dict):
    assert anls_score(d, d, return_gt=True) == (approx(1.0), d)


def test_anls_score_recursive_dicts_and_multiple_options():
    answers = {"a": {"b": ("c", "e")}}
    answer1 = {"a": {"b": "e"}}
    answer2 = {"a": {"b": "c"}}

    anls1, closest_gt1 = anls_score(answers, answer1, return_gt=True)
    anls2, closest_gt2 = anls_score(answers, answer2, return_gt=True)

    assert anls1 == anls2 == approx(1.0)
    assert closest_gt1 == answer1
    assert closest_gt2 == answer2


def test_anls_score_two_dicts_expected_multiple_options():
    answers = {"a": ("b", "t"), "c": "d"}
    answer = {"a": "t", "c": "d"}
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == answer
    assert list(closest_gt.keys()) == list(answer.keys())


@given(st.dictionaries(st.text(), st.lists(st.text())))
def test_anls_score_two_dicts_with_lists(d: dict):
    assert anls_score(d, d, return_gt=True) == (approx(1.0), d)


def test_anls_score_two_different_dict_values():
    answers = {"a": "b", "c": "d"}
    answer = {"a": "b", "c": "e"}
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(0.5)
    assert closest_gt == {"a": "b", "c": "d"}


def test_anls_score_two_different_dict_keys():
    answers = {"a": "b", "c": "d"}
    answer = {"a": "b", "e": "d"}
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(1.0 / 3.0)
    assert closest_gt == {"a": "b", "c": "d", "e": None}
    assert list(closest_gt.keys()) == ["a", "c", "e"]


@given(st.lists(st.dictionaries(st.text(), st.text())))
def test_anls_score_list_of_dict(lst: list):
    """Should find the best match if the items are dicts."""
    gt = lst.copy()
    pred = lst.copy()
    random.shuffle(pred)
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == pred
    for g, p in zip(closest_gt, pred):
        assert list(g.keys()) == list(p.keys())


def test_anls_score_line_items():
    """Complex structure with line items."""
    gt = {"k1": 1, "k2": "mock", "line_items": [{"a": "b", "c": "d"}, {"e": "f"}]}
    pred = {"k1": 1, "k2": "kcom", "line_items": [{"e": "f"}, {"a": "y", "c": "d"}]}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(3 / 5)  # fair average regardless of depth
    assert closest_gt == {
        "k1": 1,
        "k2": "mock",
        "line_items": [{"e": "f"}, {"a": "b", "c": "d"}],
    }


def test_anls_dict_list():
    gt = {"a": ["a", "b", "c"]}
    pred = {"a": ["b", "c"]}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(2 / 3)  # fair average regardless of depth
    assert closest_gt == {"a": ["b", "c", "a"]}  # correct preds first in list


def test_anls_dict_list_but_returns_scalar():
    gt = {"a": ["a", "b", "c"]}
    pred = {"a": "b"}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(0.0)
    assert closest_gt == gt


def test_anls_dict_list_different_options():
    gt = {"a": ("David Peer", "Peer David")}
    pred = {"a": "David Peer"}
    anls, closest_gt = anls_score(gt, pred, return_gt=True)

    assert anls == approx(1)
    assert closest_gt == {"a": "David Peer"}


def test_anls_more_hallucinated_keys_should_lower_score():
    gt = {"a": "b"}
    pred1 = {"a": "b", "c": "d"}
    pred2 = {"a": "b", "c": "d", "e": "f", "g": "h"}

    anls1, closest_gt_1 = anls_score(gt, pred1, return_gt=True)
    anls2, closest_gt_2 = anls_score(gt, pred2, return_gt=True)

    assert 1.0 > anls1 > anls2 > 0.0
    # Hallucinated keys are added as None-valued keys to the closest_gt
    # (minimal diff to the pred)
    assert closest_gt_1 == {"a": "b", "c": None}
    assert closest_gt_2 == {"a": "b", "c": None, "e": None, "g": None}
    assert list(closest_gt_1.keys()) == ["a", "c"]
    assert list(closest_gt_2.keys()) == ["a", "c", "e", "g"]


def test_anls_more_missed_keys_should_lower_score():
    gt = {"a": "b", "c": "d", "e": "f", "g": "h"}
    pred1 = {"a": "b", "c": "d"}
    pred2 = {"a": "b"}

    anls1, closest_gt_1 = anls_score(gt, pred1, return_gt=True)
    anls2, closest_gt_2 = anls_score(gt, pred2, return_gt=True)

    assert 1.0 > anls1 > anls2 > 0.0
    assert closest_gt_1 == gt
    assert closest_gt_2 == gt
    assert list(closest_gt_1.keys()) == list(gt.keys())
    assert list(closest_gt_2.keys()) == list(gt.keys())


### No tuples in preds ###
@given(
    st.recursive(
        st.none() | st.booleans() | st.text() | st.floats() | st.integers(),
        lambda children: st.tuples(children)
        | st.lists(children)
        | st.dictionaries(st.text(), children),
        max_leaves=10,
    ).filter(has_tuple)
)
def test_anls_tuple_in_pred_should_fail(obj):
    with pytest.raises(ValueError, match=".*Tuples are reserved.*"):
        ANLSTree.make_tree(obj, is_gt=False)


### Arbitrary objects ###
@given(
    st.recursive(
        st.none() | st.booleans() | st.text() | st.floats() | st.integers(),
        lambda children: st.lists(children) | st.dictionaries(st.text(), children),
        max_leaves=10,
    )
)
def test_arbitrary_equal_objects(obj):
    assert anls_score(obj, obj, return_gt=True) == (approx(1.0), obj)


### ST-VQA backward compatibility ###
@given(
    st.lists(st.text(), min_size=1),
    st.text(),
)
def test_anls_score_classical_qa_dataset_should_log(gt: list, pred: str):
    gt = gt + [pred]
    random.shuffle(gt)
    with pytest.warns(UserWarning, match="compatibility mode"):
        assert anls_score(gt, pred, return_gt=True) == (approx(1.0), pred)


def test_classical_qa_dataset():
    answers = [
        "st. louis children's hospital",
        "st. louis childrens hospital",
        "st. louis children hospital",
    ]
    answer = "St. Louis Children's Hospital"
    anls, closest_gt = anls_score(answers, answer, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == "st. louis children's hospital"


def test_empty_response_penalizes_FNs_only():
    answer = {
        "jurisdiction": "California",
        "party": "Ray_M._Dolby",
    }
    gt = {
        "jurisdiction": None,
        "party": None,
        "effective_date": None,
        "term": None,
    }
    anls, closest_gt = anls_score(gt, answer, return_gt=True)

    assert anls == approx(0.5)
    assert closest_gt == gt
    assert list(closest_gt.keys()) == list(gt.keys())


def test_missing_key_is_interpreted_as_none():
    answer = {
        "name": "david",
    }
    gt = {
        "name": "david",
        "company": None,
    }
    anls, closest_gt = anls_score(gt, answer, return_gt=True)

    assert anls == approx(1.0)
    assert closest_gt == gt
    assert list(closest_gt.keys()) == list(gt.keys())


#
# Cases from paper
#
def test_paper_string_correct():
    gt = "Hello World"
    answer = "Hello World"
    anls = anls_score(gt, answer)
    assert anls == approx(1.0)


def test_paper_string_typo():
    gt = "Hello World"
    answer = "Hello Wrold"
    anls = anls_score(gt, answer)

    anls = round(anls, 2)
    assert anls == approx(0.82)


def test_paper_string_wrong_answer():
    gt = "Hello World"
    answer = "How are you?"
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_paper_string_hallucination():
    gt = None
    answer = "How are you?"
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_paper_one_of_n():
    gt = tuple(["Hello", "World"])
    answer = "Hello"
    anls = anls_score(gt, answer)

    assert anls == approx(1.0)


def test_paper_one_of_n_typo():
    gt = tuple(["Hello", "World"])
    answer = "Wolrd"
    anls = anls_score(gt, answer)

    assert anls == approx(0.6)


def test_paper_expected_string():
    gt = "Hello World"
    answer = list(["Hello", "World"])
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_paper_list_correct():
    gt = list(["Hello", "World"])
    answer = list(["World", "Hello"])
    anls = anls_score(gt, answer)

    assert anls == approx(1.0)


def test_paper_list_missing_element():
    gt = list(["Hello", "World"])
    answer = list(["Hello"])
    anls = anls_score(gt, answer)

    assert anls == approx(0.5)


def test_paper_two_dicts():
    gt = {"a": "Hello", "b": "World"}
    answer = {"b": "World", "a": "Hello"}
    anls = anls_score(gt, answer)

    assert anls == approx(1.0)


def test_paper_two_dicts_one_missing_key():
    gt = {"a": "Hello", "b": "World"}
    answer = {"a": "Hello"}
    anls = anls_score(gt, answer)

    assert anls == approx(0.5)


def test_paper_two_dicts_hallucinating():
    gt = {"a": "Hello", "b": "World"}
    answer = {"b": "World", "a": "Hello", "c": "Great"}
    anls = anls_score(gt, answer)
    anls = round(anls, 2)

    assert anls == approx(0.67)


def test_paper_complex_object():
    gt = {"a": "Hello", "b": ["w", "r", "l", "d"]}
    answer = {"a": "Hello", "b": ["w", "r", "d"]}
    anls = anls_score(gt, answer)

    assert anls == approx(0.8)


def test_paper_edge_list_implicitly_casted():
    gt = list(["Hello", "World"])
    answer = "World"
    anls = anls_score(gt, answer)

    assert anls == approx(1.0)


def test_paper_edge_numbers():
    gt = 0.2
    answer = 0.199999
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_paper_edge_date_formats():
    gt = "31.12.2023"
    answer = "31. Dec 2023"
    anls = anls_score(gt, answer)
    anls = round(anls, 2)

    assert anls == approx(0.58)


def test_paper_unanswerable_wrong_answer():
    gt = "Yesterday"
    answer = "Last Week"
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_paper_unanswerable_no_answer():
    gt = "Yesterday"
    answer = None
    anls = anls_score(gt, answer)

    assert anls == approx(0.0)


def test_answer_dict_with_additional_nones_is_ignored():
    gt = {}
    answer = {
        "string": "",
        "dict": {},
        "list": [],
        "none": None,
        "bool": False,
    }

    anls = anls_score(gt, answer)
    assert anls == approx(0.0)
