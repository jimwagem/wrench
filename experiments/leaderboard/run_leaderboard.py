import logging

from wrench.classification import WeaSEL
from wrench.labelmodel import Snorkel, MajorityVoting
from wrench.labelmodel import FlyingSquid
from wrench.endmodel import EndClassifierModel
from wrench.leaderboard import ModelWrapper, make_leaderboard

N_STEPS=2000

def snorkel_model(dataset_name):
    params = dict(
        end_model = dict(
            batch_size=128,
            test_batch_size=512,
            n_steps=N_STEPS,
            backbone='MLP',
            optimizer='Adam',
            optimizer_lr=1e-2,
            optimizer_weight_decay=0.0,
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

def weasel_model(dataset_name):
    params = dict(
        temperature=1.0,
        dropout=0.3,
        hidden_size=100,

        batch_size=16,
        real_batch_size=8,
        test_batch_size=128,
        n_steps=N_STEPS,
        grad_norm=1.0,

        backbone='MLP',
        # backbone='BERT',
        backbone_model_name='MLP',
        backbone_fine_tune_layers=-1,  # fine  tune all
        optimizer='AdamW',
        optimizer_lr=5e-5,
        optimizer_weight_decay=0.0,
    )
    model = ModelWrapper(
        model_func=lambda : WeaSEL(
            **params
        ),
        name="MLP_weasel"
    )
    return model, params

def flying_squid(dataset_name, solve_method='mean'):
    fit_args = dict(label_model={'solve_method': solve_method})
    params = dict(
        batch_size=128,
        test_batch_size=512,
        n_steps=N_STEPS,
        backbone='MLP',
        optimizer='Adam',
        optimizer_lr=1e-2,
        optimizer_weight_decay=0.0,
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

def ground_truth(dataset_name):
    fit_args=dict(
        train_type = 'ground_truth'
    )
    params = dict(
        batch_size=128,
        test_batch_size=512,
        n_steps=N_STEPS,
        backbone='MLP',
        optimizer='Adam',
        optimizer_lr=1e-2,
        optimizer_weight_decay=0.0
    )
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        fit_args=fit_args,
        name='Ground_truth_MLP'
    )
    return model, params

def supervised_validation(dataset_name):
    fit_args=dict(
        train_type = 'validation'
    )
    params = dict(
        batch_size=128,
        test_batch_size=512,
        n_steps=N_STEPS,
        backbone='MLP',
        optimizer='Adam',
        optimizer_lr=1e-2,
        optimizer_weight_decay=0.0
    )
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        fit_args=fit_args,
        name='supervised_validation_MLP'
    )
    return model, params

def majority_vote(dataset_name):
    params = dict(
        batch_size=128,
        test_batch_size=512,
        n_steps=N_STEPS,
        backbone='MLP',
        optimizer='Adam',
        optimizer_lr=1e-2,
        optimizer_weight_decay=0.0
    )
    model = ModelWrapper(
        model_func=lambda: EndClassifierModel(
            **params
        ),
        label_model_func=lambda: MajorityVoting(),
        name = f'2stage_MLP_MV'
    )
    return model, params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Hyperparams
    device = 'cpu'
    datasets = ['profteacher', 'imdb', 'imdb_12', 'amazon', 'crowdsourcing']
    # datasets = [ 'imdb']
    binary_metrics = ['acc', 'auc', 'f1_binary', 'mcc'] #, 'f1']
    multi_metrics = ['acc', 'f1_macro', 'mcc']

    # Params
    model_param_pairs = [
        ground_truth(''),
        supervised_validation(''),
        snorkel_model(''),
        flying_squid('','triplet_median'),
        flying_squid('','triplet_mean'),
        majority_vote(''),
        weasel_model('')
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
        # log_file='../../logfile.csv',
        seed_range=3,
    )
    print(results)
