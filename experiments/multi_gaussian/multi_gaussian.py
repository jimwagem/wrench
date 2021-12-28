import logging
import os

import torch
import numpy as np
import pandas as pd

from wrench.classification import WeaSEL
from wrench.synthetic import MultiGaussian
from wrench.labelmodel import FlyingSquid
from wrench.utils import set_seed
from wrench.leaderboard import ModelWrapper
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import Snorkel


def compare_weights(model, test_data, model_name, n_good_lfs=5):
    if model_name.startswith('weasel'):
        w_list = [
            model.extract_weights(x)
            for x in model._init_valid_dataloader(test_data)
        ]
        weights = np.mean(w_list, axis=0)
    elif model_name == 'snorkel':
        weights = model.label_model.model.get_weights()
    elif model_name == 'flyingsquid':
        w_list = [
            fs.estimated_accuracies()
            for fs in model.label_model.model
        ]
        weights = np.mean(w_list, axis=0)
    else:
        raise ValueError(f"model_name '{model_name}' not understood")

    good_mean = np.mean(weights[:n_good_lfs])
    good_std = np.std(weights[:n_good_lfs])
    if len(weights) == n_good_lfs:
        bad_mean = np.nan
        bad_std = np.nan
    else:
        bad_mean = np.mean(weights[n_good_lfs:])
        bad_std = np.std(weights[n_good_lfs:])
    total_mean = np.mean(weights)
    total_std = np.std(weights)
    return good_mean, good_std, bad_mean, bad_std, total_mean, total_std


def write_result_pandas(file_name, result):
    is_first = not os.path.exists(file_name)
    mode = 'a' if not is_first else 'w'
    df = pd.DataFrame([result.values()], columns=result.keys())
    df.to_csv(file_name, mode=mode, index=False, header=is_first)


def generate_datasets(lf_type, n_lfs, n_good_lfs=5):
    # Set type of labeling function
    n_bad_lfs = 0
    n_random_lfs = 0
    n_constant_lfs = 0
    if lf_type == "bad":
        n_bad_lfs = n_lfs
    elif lf_type == "random":
        n_random_lfs = n_lfs
    elif lf_type == "constant":
        n_constant_lfs = n_lfs
    else:
        raise ValueError("lf_type unknown")

    mg = MultiGaussian(
        n_features=8,
        n_visible_features=7,
        n_good_lfs=n_good_lfs,
        n_bad_lfs=n_bad_lfs,
        n_random_lfs=n_random_lfs,
        n_constant_lfs=n_constant_lfs,
        n_class=5,
        sample_low=-5,
        sample_high=5
    )
    train_dataset = mg.generate_split(split='train', n_data=10000)
    train_dataset.labels = None
    valid_dataset = mg.generate_split(split='valid', n_data=1000)
    test_dataset = mg.generate_split(split='test', n_data=1000)
    return train_dataset, valid_dataset, test_dataset


def instantiate_model(model_name : str):
    if model_name.startswith('weasel'):
        model = WeaSEL(
            temperature=1.0,
            dropout=0.3,
            hidden_size=100,
            batch_size=64,
            real_batch_size=8,
            test_batch_size=128,
            n_steps=10000,
            grad_norm=1.0,
            backbone='MLP',
            # backbone='BERT',
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0,
        )
    elif model_name == 'flyingsquid':
        model = ModelWrapper(
            model_func=lambda: EndClassifierModel(
                batch_size=128,
                test_batch_size=512,
                n_steps=1000,
                backbone='MLP',
                optimizer='Adam',
                optimizer_lr=1e-2,
                optimizer_weight_decay=0.0,
            ),
            label_model_func=lambda: FlyingSquid(),
            name='2stage_MLP_flyingsquid',
        )
    elif model_name == 'snorkel':
        model = ModelWrapper(
            model_func=lambda: EndClassifierModel(
                batch_size=128,
                test_batch_size=512,
                n_steps=1000,
                backbone='MLP',
                optimizer='Adam',
                optimizer_lr=1e-2,
                optimizer_weight_decay=0.0,
            ),
            label_model_func=lambda: Snorkel(
                lr=0.01,
                l2=0.0,
                n_epochs=10,
            ),
            name='2stage_MLP_snorkel',
        )
    else:
        raise ValueError(f"model_name '{model_name}' not understood")

    return model


def printd(d):
    return ",".join([f"{key}='{value}'" for key, value in d.items()])


if __name__ == '__main__':
    device = torch.device('cpu')
    seeds = 20

    n_good_lfs = 5
    models = [
        'weasel_nofeats',
        'weasel',
        'snorkel',
        'flyingsquid',
    ]
    lf_types = [
        'bad',
        'random',
        'constant',
    ]
    metrics = [
        'acc',
        'mcc',
    ]

    # Same as in Figure 2 of Weasel paper
    task_lfs = list(range(0, 10))
    # Same as in Figure 4 of Weasel paper
    task_lfs += [10, 15, 20, 25, 50, 75, 100]

    # Show INFO, WARNING and ERROR logging messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Results file name (tip: change to split results per model etc.)
    file_name = f'./results/multi_gauss_new.csv'

    for model_name in models:
        for lf_type in lf_types:
            for n_lfs in task_lfs:
                for i in range(seeds):
                    run_result = {
                        'model_name': model_name,
                        'lf_type': lf_type,
                        'n_good_lfs': n_good_lfs,
                        'n_lfs': n_lfs,
                        'run_id': i,
                    }
                    logger.info(f"experiment with {printd(run_result)}")
                    set_seed(i)

                    train_dataset, valid_dataset, test_dataset = generate_datasets(
                        lf_type, n_lfs, n_good_lfs=n_good_lfs
                    )
                    fit_args = {}
                    if model_name == 'weasel_nofeats':
                        fit_args['use_encoder_features'] = False

                    logger.info("Instantiate and train model")
                    model = instantiate_model(model_name)
                    model.fit(
                        train_dataset,
                        valid_dataset,
                        evaluation_step=10,
                        metric='acc',
                        patience=100,
                        device=device,
                        verbose=False,
                        **fit_args,
                    )

                    logger.info("Get results from trained model")
                    # Get weights from model
                    (
                        run_result['good_mean'],
                        run_result['good_std'],
                        run_result['bad_mean'],
                        run_result['bad_std'],
                        run_result['total_mean'],
                        run_result['total_std'],
                    ) = compare_weights(
                        model, test_data=test_dataset, model_name=model_name
                    )

                    # Obtain metrics
                    test_results = model.test(test_dataset, metrics)
                    for metric, score in zip(metrics, test_results):
                        run_result[metric] = score

                    logger.info('Writing results to file')
                    write_result_pandas(file_name, run_result)
