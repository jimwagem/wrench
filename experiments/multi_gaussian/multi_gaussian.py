import torch
import numpy as np
from wrench.classification import WeaSEL
from wrench.synthetic import MultiGaussian
from wrench.labelmodel import FlyingSquid
from wrench.utils import set_seed
from wrench.leaderboard import ModelWrapper
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import Snorkel
import csv
import matplotlib.pyplot as plt

def compare_weights(model, test_data, model_name, n_good_lfs=5):
    if model_name == 'weasel':
        w_list = []
        for x in model._init_valid_dataloader(test_data):
            w_list.appned(model.extract_weights(x))
        weights = np.mean(w_list)
    elif model_name == 'snorkel':
        weights = model.labelmodel.get_weights()
    elif model_name == 'flyingsquid':
        weights = model.labelmodel.model.probability_values
    return (np.mean(weights[:n_good_lfs]), np.mean(weights[n_good_lfs:]))


def write_results(log_file, lf_type, acc, std, n_lfs):
    with open(log_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([lf_type, n_lfs, acc, std])

if __name__=='__main__':
    device = torch.device('cpu')
    seeds = 20
    
    model_name='weasel'
    lf_type='random'
    log_weight=True

    for n_lfs in range(0, 10):
        print(f'testing {n_lfs}')
        seed_results=[]
        for i in range(seeds):
            set_seed(i)

            # Set type of labeling function
            n_bad_lfs=0,
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
                n_constant_lfs=n_random_lfs,
                n_class=5,
                sample_low=-5,
                sample_high=5)
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
            seed_results.append(model.test(test_dataset, 'acc'))
        print(np.mean(seed_results))
        print(np.std(seed_results))
        write_results(f'./results/multi_gauss_{model_name}.csv', '', np.mean(seed_results), np.std(seed_results), n_lfs)
    
    

