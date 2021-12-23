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
import csv


def compare_weights(model, test_data, model_name, n_good_lfs=5):
    if model_name == 'weasel':
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
        weights = np.mean(w_list, axis = 0)
    else:
        raise ValueError("model_name not understood")
    good_mean = np.mean(weights[:n_good_lfs])
    if len(weights) == n_good_lfs:
        bad_mean = np.nan
    else:
        bad_mean = np.mean(weights[n_good_lfs:])
    return (good_mean, bad_mean)


def write_results(log_file, *args):
    with open(log_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(args)


def write_results_pandas(file_name, rows, columnsm, is_first):
    df = pd.DataFrame(rows, columns=columns)
    if is_first:
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, mode='a', index=False, header=False)


if __name__ == '__main__':
    device = torch.device('cpu')
    seeds = 20
    
    model_name='weasel'
    lf_types=['bad', 'random', 'constant']
    log_weights=False

    # Same as in Figure 2 of Weasel paper
    lfs2 = [10, 15, 20, 25, 50, 75, 100]
    print(lfs2)
    # Same as in Figure 4 of Weasel paper
    # lfs = lfs2 + [15, 20, 25, 50, 75, 100]
    is_first = False
    for lf_type in lf_types:
        print(f'testing lf type: {lf_type}')
        rows = []
        for n_lfs in lfs2:
            print(f'testing {n_lfs}')
            if log_weights:
                good_weights = []
                bad_weights = []
            for i in range(seeds):
                set_seed(i)
                seed_results = []

                # Set type of labeling function
                n_bad_lfs=0
                n_random_lfs=0
                n_constant_lfs=0
                if lf_type=="bad":
                    n_bad_lfs=n_lfs
                elif lf_type=="random":
                    n_random_lfs=n_lfs
                elif lf_type=="constant":
                    n_constant_lfs=n_lfs
                mg = MultiGaussian(n_features=8, n_visible_features=7,
                    n_good_lfs=5,
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
                if model_name=='weasel':
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
                elif model_name=='flyingsquid':
                    model = ModelWrapper(
                        model_func=lambda : EndClassifierModel(
                            batch_size=128,
                            test_batch_size=512,
                            n_steps=1000,
                            backbone='MLP',
                            optimizer='Adam',
                            optimizer_lr=1e-2,
                            optimizer_weight_decay=0.0,
                        ),
                        label_model_func=lambda: FlyingSquid(),
                        name = '2stage_MLP_flyingsquid'
                    )
                elif model_name=='snorkel':
                    model = ModelWrapper(
                        model_func=lambda : EndClassifierModel(
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
                            n_epochs=10
                        ),
                        name = '2stage_MLP_snorkel'
                    )
                else:
                    raise ValueError("'model_name' not understood")
                # model.fit(
                #     train_dataset,
                #     valid_dataset,
                #     10,
                #     metric='acc',
                #     patience=100,
                #     device=device,
                #     verbose=False
                #     )
                model.fit(
                    train_dataset,
                    valid_dataset,
                    evaluation_step=10,
                    metric='acc',
                    patience=100,
                    device=device,
                    verbose=False
                )
                if log_weights:
                    good_mean, bad_mean = compare_weights(model, test_data=test_dataset, model_name=model_name)
                    good_weights.append(good_mean)
                    bad_weights.append(bad_mean)
                    rows.append(
                        (lf_type, n_lfs, i, good_mean, bad_mean)
                    )
                else:
                    seed_results.append(model.test(test_dataset, 'acc'))
                
        if log_weights:
            columns = ['lf_type', 'n_lfs', 'seed', 'good_mean', 'bad_mean']
            write_results_pandas(f'./results/multi_gauss_weights_{model_name}.csv', rows, columns, is_first=is_first)
        else:
            print(np.mean(seed_results))
            print(np.std(seed_results))
            write_results(f'./results/multi_gauss_{model_name}.csv', '', n_lfs, np.mean(seed_results), np.std(seed_results))
        is_first = False

        # col1 = [2,3,4]
        # col2 = ['a','b','c']
        # df= pd.DataFrame(data=[col1, col2], columns=['col1', 'col2'])
        # df.groupby('model_name')['accuracy'].mean()
        # df.to_csv('hello.csv')
