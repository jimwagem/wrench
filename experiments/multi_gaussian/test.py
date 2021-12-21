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

    mg = MultiGaussian(n_features=8, n_visible_features=7,
        n_good_lfs=5,
        n_bad_lfs=5,
        n_random_lfs=0,
        n_constant_lfs=0,
        n_class=5,
        sample_low=-5,
        sample_high=5)
    train_dataset = mg.generate_split(split='train', n_data=100)
    train_dataset.labels = None
    valid_dataset = mg.generate_split(split='valid', n_data=100)
    test_dataset = mg.generate_split(split='test', n_data=100)
    
    model = WeaSEL(
        temperature=1.0,
        dropout=0.3,
        hidden_size=100,

        batch_size=64,
        real_batch_size=8,
        test_batch_size=128,
        n_steps=100,
        grad_norm=1.0,

        backbone='MLP',
        # backbone='BERT',
        optimizer='AdamW',
        optimizer_lr=5e-5,
        optimizer_weight_decay=0.0,
    )
    
    model.fit(
        train_dataset,
        valid_dataset,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device,
        verbose=False
    )
    compare_weights(model, test_dataset)
            
    
    

