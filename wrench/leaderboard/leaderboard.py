# import logging
import torch
import numpy as np
import csv
from ..dataset import load_dataset
from ..utils import set_seed
# from wrench.classification import WeaSEL
# from wrench.labelmodel import Snorkel
# from wrench.endmodel import EndClassifierModel

class ModelWrapper():
    """Model wrapper such that we can compare 2stage and end2end models.
    These"""
    def __init__(self, model_func, name, label_model_func=None):
        self.model_func=model_func
        self.label_model_func=label_model_func
        self.name = name
        self.reset()
    
    def reset(self):
        self.model=self.model_func()
        if self.label_model_func is not None:
            self.label_model=self.label_model_func()
        else:
            self.label_model=None

    def fit(self, train_data, valid_data, metric, evaluation_step=10, patience=100, device='cpu'):
        if self.label_model is not None:
            # 2stage model
            self.label_model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data
                )
            train_data = train_data.get_covered_subset()
            soft_labels = self.label_model.predict_proba(train_data)
            self.model.fit(
                dataset_train=train_data,
                y_train=soft_labels,
                dataset_valid=valid_data,
                evaluation_step=evaluation_step,
                metric=metric,
                patience=patience,
                device=device
            )
        else:
          self.model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data,
                evaluation_step=evaluation_step,
                metric=metric,
                patience=patience,
                device=device
            )

    def test(self, metrics, test_data):
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


def make_leaderboard(models, datasets, metrics, dataset_path='../data/', device='cpu',
                     save_dir=None, log_file=None, seed_range=None):
    """Trains each model on each datasets and evaluates the different metrics."""
    results = []
    if seed_range is None:
        seed_range = np.arange(1)
    if isinstance(seed_range, int):
        seed_range = np.arange(seed_range)

    for dataset in datasets:
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            extract_fn='bert',
            cache_name='bert',
            device=device
        )
        dataset_results = []
        for model in models:
            model_results = []
            for seed in seed_range:
                set_seed(seed)
                model.reset()
                model.fit(train_data, valid_data, metrics[0], device=device)
                seed_results = model.test(metrics, test_data)
                if save_dir is not None:
                    model.save(save_dir, dataset)
                if log_file is not None:
                    model.log_csv(log_file, dataset, metrics, seed_results)
                model_results.append(seed_results)
        dataset_results.append(model_results)
        results.append(dataset_results)
    return results


# if __name__=="__main__":
#     device='cpu'
#     datasets = ['youtube', 'sms']
#     model1 = ModelWrapper(
#         model_func=lambda : WeaSEL(
#             temperature=1.0,
#             dropout=0.3,
#             hidden_size=100,

#             batch_size=16,
#             real_batch_size=8,
#             test_batch_size=128,
#             n_steps=1000,
#             grad_norm=1.0,

#             backbone='MLP',
#             # backbone='BERT',
#             backbone_model_name='MLP',
#             backbone_fine_tune_layers=-1,  # fine  tune all
#             optimizer='AdamW',
#             optimizer_lr=5e-5,
#             optimizer_weight_decay=0.0,
#         )
#     )
#     model2 = ModelWrapper(
#         model_func=lambda : EndClassifierModel(
#             batch_size=128,
#             test_batch_size=512,
#             n_steps=1000,
#             backbone='MLP',
#             optimizer='Adam',
#             optimizer_lr=1e-2,
#             optimizer_weight_decay=0.0,
#         ),
#         label_model_func=lambda: Snorkel(
#             lr=0.01,
#             l2=0.0,
#             n_epochs=10
#         )
#     )
#     models = [model1, model2]
#     metrics = ['acc']
#     print(leaderboard(models=models, datasets=datasets, metrics=metrics, dataset_path='../../datasets/'))
