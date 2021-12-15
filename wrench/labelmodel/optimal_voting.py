import logging
from typing import Any, Optional, Union

import numpy as np

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class OptimalVoting(BaseLabelModel):
    """Oracle voter that predicts the true label if it is present in the label matrix, or votes randomly otherwise.
    The performance of this voter can be used as a reference for the upper bound for voters. Makes the (common)
    assumption that the LF output is not flipped w.r.t. class label."""
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.n_class = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            n_class: Optional[int] = None,
            **kwargs: Any):
        if dataset_train.labels is None:
            raise ValueError("This voter requires the dataset ground truth labels!")

        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        self.n_class = n_class or np.max(check_weak_labels(dataset_train)) + 1

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        if dataset.labels is None:
            raise ValueError("This voter requires the dataset ground truth labels!")

        labels = np.array(dataset.labels)
        result = [labels[i] if (labels[i] in L[i, :]) else ABSTAIN for i in range(len(labels))]
        n_class = self.n_class
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            r = result[i]
            if r != ABSTAIN:
                counts[r] = 1
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p
