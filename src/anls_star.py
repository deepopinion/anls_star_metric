""" Official ANLS* metric implementation. 

Paper: ANLS* - A Universal Document Processing Metric for Generative Large Language Models
Authors: David Peer, Philemon Schöpf, Volckmar Nebendahl, Alexander Rietzler, Sebastian Stabinger
Link: ToDo

DeepOpinion, 2024
"""

import abc
import warnings
from typing import Any

from munkres import Munkres, make_cost_matrix


class ANLSTree(abc.ABC):
    THRESHOLD = 0.5  # ANLS threshold. 0.5 is a standard value.
    obj: Any
    tree: Any

    @staticmethod
    def make_tree(obj, *, is_gt: bool) -> "ANLSTree":
        """Make an ANLS tree from a complex object.
        Args:
            obj: The object to make a tree from.
            is_gt: Whether the object is a ground truth object. Ground truths are allowed to have multiple valid options via tuples. Predictions are not allowed to have tuples.
        Returns: Parent node of the ANLS tree.
        """
        if isinstance(obj, tuple):
            return ANLSTuple(obj, is_gt=is_gt)
        elif isinstance(obj, list):
            return ANLSList(obj, is_gt=is_gt)
        elif isinstance(obj, dict):
            return ANLSDict(obj, is_gt=is_gt)
        elif obj is None:
            return ANLSNone()
        if isinstance(obj, (str, float, int, bool)):
            return ANLSLeaf(obj)
        else:
            raise ValueError(
                f"Found unsupported type {type(obj)} for {obj} while creating ANLS tree"
            )

    def anls(self, other: "ANLSTree") -> float:
        nls_list = self.nls_list(other)
        length = self.pairwise_len(other)
        return (sum(nls_list) / length) if length > 0 else 1.0

    def __str__(self) -> str:
        return f"ANLSTree({repr(self.obj)})"

    def __repr__(self) -> str:
        return f"ANLSTree({repr(self.obj)})"

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def pairwise_len(self, other: "ANLSTree") -> int:
        pass

    @abc.abstractmethod
    def nls_list(self, other: "ANLSTree") -> list[float]:
        pass


class ANLSTuple(ANLSTree):
    def __init__(self, obj, is_gt: bool):
        if not isinstance(obj, tuple):
            raise ValueError(f"ANLSTuple expects a tuple, got {type(obj)}")
        if not is_gt:
            raise ValueError(
                "Tuples are reserved for 1-of-n ground truths. Use lists as containers in predictions."
            )
        if len(obj) == 0:
            raise ValueError("Expected at least 1 valid ground truth option")
        self.obj = obj
        self.tree: tuple[ANLSTree, ...] = tuple(
            ANLSTree.make_tree(x, is_gt=is_gt) for x in obj
        )

    def __len__(self):
        return max(len(x) for x in self.tree)

    def _choose_best_item(self, other):
        candidate_nlss: list[list[float]] = []
        lengths: list[int] = []
        for gt in self.tree:
            candidate_nlss.append(gt.nls_list(other))
            lengths.append(gt.pairwise_len(other))
        # Select the best matching choice

        def avg_nls(tuple_):
            nls_list, length = tuple_
            return (sum(nls_list) / length) if length > 0 else 1.0

        best_nls, best_length = max(zip(candidate_nlss, lengths), key=avg_nls)
        return best_nls, best_length

    def pairwise_len(self, other):
        best_nls, best_length = self._choose_best_item(other)
        return best_length

    def nls_list(self, other):
        best_nls, best_length = self._choose_best_item(other)
        return best_nls


class ANLSList(ANLSTree):
    def __init__(self, obj, is_gt: bool):
        if not isinstance(obj, list):
            raise ValueError(f"ANLSList expects a list, got {type(obj)}")
        self.obj = obj
        self.tree: list[ANLSTree] = [ANLSTree.make_tree(x, is_gt=is_gt) for x in obj]

    def __len__(self):
        return sum(len(x) for x in self.tree)

    def _hungarian(self, other: "ANLSList"):
        mat: list[list[list[float]]] = []
        avg_mat: list[list[float]] = []
        for gt in self.tree:
            row = []
            avg_row = []
            for pred in other.tree:
                nls_list = gt.nls_list(pred)
                length = gt.pairwise_len(pred)
                row.append(nls_list)
                avg_row.append((sum(nls_list) / length) if length > 0 else 1.0)
            mat.append(row)
            avg_mat.append(avg_row)

        # Check for empty lists - Munkres fails on empty
        if len(mat) == 0 or len(mat[0]) == 0:
            return mat, []

        # Run Hungarian algorithm
        m_cost_matrix = make_cost_matrix(avg_mat)
        indexes = Munkres().compute(m_cost_matrix)
        return mat, indexes

    def pairwise_len(self, other):
        if not isinstance(other, ANLSList):
            return max(len(self), len(other))
        _, indexes = self._hungarian(other)

        not_selected_self = {*range(len(self.tree))} - {row for row, _ in indexes}
        not_selected_other = {*range(len(other.tree))} - {col for _, col in indexes}

        pwl = sum(self.tree[row].pairwise_len(other.tree[col]) for row, col in indexes)
        pwl += sum(len(self.tree[i]) for i in not_selected_self)
        pwl += sum(len(other.tree[j]) for j in not_selected_other)
        return pwl

    def nls_list(self, other: ANLSTree):
        if not isinstance(other, ANLSList):
            return [0.0]

        mat, indexes = self._hungarian(other)
        values = [mat[row][column] for row, column in indexes]
        values = [item for sublist in values for item in sublist]

        return values


class ANLSDict(ANLSTree):
    def __init__(self, obj, is_gt: bool):
        if not isinstance(obj, dict):
            raise ValueError(f"ANLSDict expects a dict, got {type(obj)}")
        self.obj = obj
        self.tree: dict[Any, ANLSTree] = {
            k: ANLSTree.make_tree(v, is_gt=is_gt) for k, v in obj.items()
        }

    def __len__(self):
        return sum(len(x) for x in self.tree.values())

    def pairwise_len(self, other):
        if not isinstance(other, ANLSDict):
            return max(len(self), len(other))
        pwl = 0
        for k in self.tree.keys() | other.tree.keys():
            self_value = self.tree.get(k, ANLSNone())
            other_value = other.tree.get(k, ANLSNone())
            pwl += self_value.pairwise_len(other_value)
        return pwl

    def nls_list(self, other):
        if not isinstance(other, ANLSDict):
            return [0.0]

        nlss = []
        for k in self.tree.keys() | other.tree.keys():
            self_value = self.tree.get(k, ANLSNone())
            other_value = other.tree.get(k, ANLSNone())

            is_hallucinated_none_key = (
                k not in self.tree 
                and k in other.tree 
                and ANLSNone.check_if_none(other_value.obj)
            )
            if is_hallucinated_none_key:
                continue

            nls_list = self_value.nls_list(other_value)
            nlss.extend(nls_list)
        return nlss


class ANLSNone(ANLSTree):
    def __init__(self):
        self.obj = None

    def __len__(self):
        return 1

    def pairwise_len(self, other):
        return max(len(self), len(other))

    def nls_list(self, other):
        return [1.0 if self.check_if_none(other.obj) else 0.0]

    @classmethod
    def check_if_none(cls, value):
        return isinstance(value, ANLSNone) or value in (None, {}, [], "")


class ANLSLeaf(ANLSTree):
    def __init__(self, obj):
        if not isinstance(obj, (str, float, int, bool)):
            raise ValueError(f"Leaf must be a primitive type, got {type(obj)}")
        self.obj = obj

    def __len__(self):
        return 1

    def pairwise_len(self, other):
        return max(len(self), len(other))

    def nls_list(self, other):
        if not isinstance(other, ANLSLeaf):
            # Type mismatch, so the ANLS is 0. But we still calculate the length.
            return [0.0]

        this_str = " ".join(str(self.obj).strip().lower().split())
        other_str = " ".join(str(other.obj).strip().lower().split())

        dist = self._levenshtein_distance(this_str, other_str)
        str_length = max(len(this_str), len(other_str))
        value = 0.0 if str_length == 0 else float(dist) / float(str_length)

        question_result = 1 - value

        if question_result < self.THRESHOLD:
            question_result = 0.0

        return [question_result]

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]


def anls_score(gt, pred):
    """Run ANLS on a ground truth and prediction object. The returned score is a value between 0 and 1, where 1 is the best possible score. For further information on the ANLS metric and the types see https://arxiv.org/abs/2402.03848

    Types of gt and pred:
        - `String`: To compare strings against each other using the normalized Levenshtein similarity.
        - `None`: Sometimes questions are not answerable. With this type it can be checked, whether the model does not answer. Any answer other than None will be penalized.
        - `Tuple`: To compare a list of possible answers against each other. This is useful for tasks where multiple answers are possible. The closest match is used to compute the score. This is also provided by the classical ANLS score.
        - `List`: Sometimes it is required to information in the form of lists from a document. For example, extracting all purchased items found in an invoice. While the order is not important, the list should contain all items. Note that the same item can occur multiple times in lists. Hungarian matching is used to compare the ground truth and the predicted list against each other. Both missing elements as well as hallucinated elements are penalized.
        - `Dict`: For document information extraction it is usually required to extract key-value pairs. For example, when extracting the date and total value from an invoice. Missing keys as well as hallucinated keys are penalized.

    Note: The edge case of a ground truth being a list of strings and the prediction being a string is supported for compatibility with VQA-like datasets. In this case the list is converted implicitly to a tuple.

    Args:
        gt: The ground truth object. Can be a string, list, tuple, dict, or any combination of those. See type descriptions above.
        pred: The prediction object - usually the output of the model. Can be a string, list, tuple, dict, or any combination of those. See type descriptions above.
    
    Returns:
        The ANLS score \in [0, 1]
    """
    
    # Convert gt list to tuple in order to be compatible with classical QA datasets
    gt_is_list_str = isinstance(gt, list) and all(isinstance(x, str) for x in gt)
    pred_is_str = isinstance(pred, str)
    is_classical_qa_datasaet = gt_is_list_str and pred_is_str
    if is_classical_qa_datasaet:
        warnings.warn(
            "Treating ground truth as a list of options. This is a compatibility mode for ST-VQA-like datasets."
        )
        gt = tuple(gt)

    gt_tree = ANLSTree.make_tree(gt, is_gt=True)
    pred_tree = ANLSTree.make_tree(pred, is_gt=False)
    anls_score = gt_tree.anls(pred_tree)

    return anls_score
