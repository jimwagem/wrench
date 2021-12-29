import logging
from pathlib import Path

import torch
import numpy as np
import csv
from ..dataset import load_dataset
from ..utils import set_seed


class ModelWrapper:
    """Model wrapper such that we can compare 2stage and end2end models.
    These"""
    def __init__(self, model_func, name, label_model_func=None):
        self.model = None
        self.label_model = None
        self.model_func = model_func
        self.label_model_func = label_model_func
        self.name = name
        self.reset()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def reset(self):
        self.model = self.model_func()
        if self.label_model_func is not None:
            self.label_model = self.label_model_func()
        else:
            self.label_model = None

    def fit(self, train_data, valid_data, metric, evaluation_step=10, patience=100, device='cpu', verbose=True, **fit_kwargs):
        kwargs = {}
        # never provide training labels
        if train_data.labels is not None:
            # train_data.labels = None
            self.logger.warning("training labels should not be provided to label model")

        if self.label_model is not None:
            # 2stage model

            self.label_model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data,
                **fit_kwargs,
            )
            train_data = train_data.get_covered_subset()
            soft_labels = self.label_model.predict_proba(train_data)
            kwargs['y_train'] = soft_labels
        self.model.fit(
            dataset_train=train_data,
            dataset_valid=valid_data,
            evaluation_step=evaluation_step,
            metric=metric,
            patience=patience,
            device=device,
            verbose=verbose,
            **kwargs,
        )

    def test(self, test_data, metrics):
        metrics = self.model.test(test_data, metrics)
        return metrics
    
    def save(self, save_dir, dataset_name):
        name = self.name + '_' + dataset_name
        if self.label_model is not None:
            self.label_model.save(save_dir + name + '.label')
        torch.save(self.model.model.state_dict(), save_dir + name + '.pt')
    
    def log_csv(self, log_file, dataset, metrics, results):
        if not isinstance(results, list):
            results = [results]
        with open(log_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for metric, result in zip(metrics, results):
                write_list = [self.name, dataset, metric, result]
                writer.writerow(write_list)


def make_leaderboard(
    models,
    datasets,
    binary_metrics,
    multi_metrics,
    dataset_path='../data/',
    device='cpu',
    save_dir=None,
    log_file=None,
    seed_range=1,
):
    """Trains each model on each datasets and evaluates the different metrics."""
    results = []
    if isinstance(seed_range, int):
        seed_range = np.arange(seed_range)

    logger = logging.getLogger(__name__)

    # Ensure directories exist
    if save_dir is not None:
        model_dir = Path(save_dir)
        model_dir.mkdir(exist_ok=True)
    if log_file is not None:
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)

    for dataset in datasets:


        logger.info(f"dataset: {dataset}")
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            extract_fn='bert',
            cache_name='bert',
            device=device
        )

        # Binary and multi class datasets use different metrics:
        binary= (np.max(test_data.labels == 1))
        if binary:
            metrics = binary_metrics
        else:
            metrics = multi_metrics
        dataset_results = []
        for model in models:
            logger.info(f"model: {model.name}")
            model_results = []
            for seed in seed_range:
                # Model training and evaluation.
                logger.info(f"run: {seed}")
                set_seed(seed)
                model.reset()
                model.fit(train_data, valid_data, metrics[0], device=device)
                seed_results = model.test(test_data, metrics)
                if save_dir is not None:
                    model.save(save_dir, dataset)
                if log_file is not None:
                    model.log_csv(log_file, dataset, metrics, seed_results)
                model_results.append(seed_results)
            dataset_results.append(model_results)
        results.append(dataset_results)
    return results
