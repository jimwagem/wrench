import logging

from wrench.classification import WeaSEL
from wrench.labelmodel import Snorkel, MajorityVoting
from wrench.labelmodel import FlyingSquid
from wrench.endmodel import EndClassifierModel
from wrench.leaderboard import ModelWrapper, make_leaderboard
from wrench.labelmodel.optimal_voting import OptimalVoting
from wrench.dataset import load_dataset
import numpy as np
N_STEPS=20000

def snorkel_model():
    params = dict(
        end_model = dict(
            batch_size=64,
            test_batch_size=64,
            n_steps=N_STEPS,
            backbone='FlexMLP',
            optimizer='Adam',
            optimizer_lr=5e-5,
            optimizer_weight_decay=7e-7,
        ),
        label_model = dict(
            lr=0.01,
            l2=0.0,
            n_epochs=10
        )
    )
    model = ModelWrapper(
        model_func=lambda : EndClassifierModel(
            **params['end_model']
        ),
        label_model_func=lambda: Snorkel(
            **params['label_model']
        ),
        name = '2stage_MLP_snorkel'
    )
    return model, params

def weasel_model(use_balance=True):
    if use_balance:
        balance='balance'
    else:
        balance='no_balance'
    params = dict(
        temperature=1.0,
        dropout=0.3,
        hidden_size=70,

        batch_size=64,
        real_batch_size=64,
        test_batch_size=64,
        n_steps=N_STEPS,
        grad_norm=1.0,

        backbone='FlexMLP',
        # backbone='BERT',
        backbone_model_name='MLP',
        backbone_fine_tune_layers=-1,  # fine  tune all
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7,
        use_balance=use_balance
    )
    model = ModelWrapper(
        model_func=lambda : WeaSEL(
            **params
        ),
        name=f"MLP_weasel_{balance}"
    )
    return model, params

def flying_squid( solve_method='mean'):
    fit_args = dict(label_model={'solve_method': solve_method})
    params = dict(
        batch_size=64,
        test_batch_size=64,
        n_steps=N_STEPS,
        backbone='FlexMLP',
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7,
    )
    model = ModelWrapper(
            model_func=lambda: EndClassifierModel(
                **params
            ),
            label_model_func=lambda: FlyingSquid(),
            name=f'2stage_MLP_flyingsquid_{solve_method}',
            fit_args=fit_args
        )
    return model, params

def ground_truth():
    fit_args=dict(
        train_type = 'ground_truth'
    )
    params = dict(
        batch_size=64,
        test_batch_size=64,
        n_steps=N_STEPS,
        backbone='FlexMLP',
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7
    )
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        fit_args=fit_args,
        name='Ground_truth_MLP'
    )
    return model, params

def supervised_validation():
    fit_args=dict(
        train_type = 'validation'
    )
    params = dict(
        batch_size=64,
        test_batch_size=64,
        n_steps=3000,
        backbone='FlexMLP',
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7
    )
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        fit_args=fit_args,
        name='supervised_validation_MLP'
    )
    return model, params

def majority_vote( hard_label=True):
    fit_args=dict(
        hard_label=hard_label
    )
    params = dict(
        batch_size=64,
        test_batch_size=64,
        n_steps=N_STEPS,
        backbone='FlexMLP',
        optimizer='Adam',
        optimizer_lr=5e-5,
        optimizer_weight_decay=7e-7
    )
    if hard_label:
        hard_label_string = 'hard'
    else:
        hard_label_string = 'soft'
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        label_model_func=lambda: MajorityVoting(),
        name = f'2stage_MLP_MV_{hard_label_string}',
        fit_args=fit_args
    )
    return model, params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Hyperparams
    device = 'cpu'
    # datasets = ['profteacher', 'imdb_12', 'imdb_136', 'amazon','crowdsourcing']
    datasets = ['crowdsourcing']
    binary_metrics = [ 'auc','acc', 'f1_binary','f1_max', 'mcc', 'ece']
    multi_metrics = ['acc', 'f1_macro', 'mcc', 'ece']

    # Params
    model_param_pairs = [
        ground_truth(),
        supervised_validation(),
        snorkel_model(),
        flying_squid('triplet_mean'),
        majority_vote(hard_label=True),
        majority_vote(hard_label=False),
        weasel_model(use_balance=False),
        weasel_model(use_balance=True)
    ]
    models = [m for m, _ in model_param_pairs]
    # Ground truth, supervised val, snorkel, flying_squid_med, flying_squid_mean, maj vote, weasel
    results = make_leaderboard(
        models=models,
        datasets=datasets,
        binary_metrics=binary_metrics,
        multi_metrics=multi_metrics,
        dataset_path='../../../datasets/',
        # save_dir='../../saved_models/',
        log_file='./results/results.csv',
        seed_range=np.arange(1,4),
        verbose=True,
        resplit_datasets=True
    )

    # Test Upper bound
    # ub_model = OptimalVoting()

