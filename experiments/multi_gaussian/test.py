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
        w_list = [
            model.extract_weights(x)
            for x in model._init_valid_dataloader(test_data)
        ]
        print(w_list)
        weights = np.mean(w_list, axis=0)
        print(weights)
    elif model_name == 'snorkel':
        weights = model.label_model.model.get_weights()
        print(weights)
    elif model_name == 'flyingsquid':
        w_list = [
            fs.estimated_accuracies()
            for fs in model.label_model.model
        ]
        print(w_list)
        weights = np.mean(w_list, axis = 0)
    else:
        raise ValueError("model_name not understood")
    return (np.mean(weights[:n_good_lfs]), np.mean(weights[n_good_lfs:]))


def write_results(log_file, lf_type, acc, std, n_lfs):
    with open(log_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([lf_type, n_lfs, acc, std])

if __name__=='__main__':
    device = torch.device('cpu')

    mg = MultiGaussian(n_features=8, n_visible_features=7,
        n_good_lfs=5,
        n_bad_lfs=0,
        n_random_lfs=10,
        n_constant_lfs=0,
        n_class=5,
        sample_low=-5,
        sample_high=5)
    train_dataset = mg.generate_split(split='train', n_data=1000)
    train_dataset.labels = None
    valid_dataset = mg.generate_split(split='valid', n_data=100)
    test_dataset = mg.generate_split(split='test', n_data=1000)
    
    model = WeaSEL(
        temperature=1.0,
        dropout=0.3,
        hidden_size=100,

        batch_size=64,
        real_batch_size=8,
        test_batch_size=128,
        n_steps=1000,
        grad_norm=1.0,

        backbone='MLP',
        # backbone='BERT',
        optimizer='AdamW',
        optimizer_lr=5e-5,
        optimizer_weight_decay=0.0,
    )
    # model = ModelWrapper(
    #                 model_func=lambda : EndClassifierModel(
    #                     batch_size=128,
    #                     test_batch_size=512,
    #                     n_steps=1000,
    #                     backbone='MLP',
    #                     optimizer='Adam',
    #                     optimizer_lr=1e-2,
    #                     optimizer_weight_decay=0.0,
    #                 ),
    #                 label_model_func=lambda: FlyingSquid(),
    #                 name = '2stage_MLP_flyingsquid'
    #             )
    # model = ModelWrapper(
    #     model_func=lambda : EndClassifierModel(
    #         batch_size=128,
    #         test_batch_size=512,
    #         n_steps=1000,
    #         backbone='MLP',
    #         optimizer='Adam',
    #         optimizer_lr=1e-2,
    #         optimizer_weight_decay=0.0,
    #     ),
    #     label_model_func=lambda: Snorkel(
    #         lr=0.01,
    #         l2=0.0,
    #         n_epochs=10
    #     ),
    #     name = '2stage_MLP_snorkel'
    # )
    model.fit(
        train_dataset,
        valid_dataset,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device,
        verbose=False
    )
    print(compare_weights(model, test_dataset, "weasel"))
            
    
    

