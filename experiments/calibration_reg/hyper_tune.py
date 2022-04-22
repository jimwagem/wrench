import logging
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from wrench.utils import set_seed
from wrench.dataset import load_dataset, resplit_dataset
from wrench.classification import WeaSEL
import pickle
import csv

if __name__=="__main__":
    device = torch.device('cpu')

    dataset_path = '../../../datasets/'
    data = 'profteacher'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert',
        cache_name='bert',
        device=device,
    )
    res_file = open('./hyper_param_results.csv', 'a+')
    csv_writer = csv.writer(res_file)
    weight_range = [2.0, 1.0, 0.5, 0.25, 0.125]
    n_seeds = 5
    # reg_types = ['L1','bhat','entropy','log_acc']
    reg_types = ['log_acc']
    # Generate results
    for weight in weight_range:
        for reg_type in reg_types:
            for seed in range(n_seeds):
                set_seed(seed)
                model = WeaSEL(
                        temperature=1.0,
                        dropout=0.3,
                        hidden_size=70,

                        batch_size=128,
                        real_batch_size=64,
                        test_batch_size=64,
                        n_steps=20000,
                        grad_norm=1.0,

                        backbone='FlexMLP',
                        # backbone='MLP',
                        # backbone='BERT',
                        backbone_model_name='MLP',
                        backbone_fine_tune_layers=-1,  # fine  tune all
                        optimizer='Adam',
                        optimizer_lr=5e-5,
                        optimizer_weight_decay=7e-7,
                        use_balance=False,
                        per_class_acc=False,
                        reg_weight=weight,
                        use_sigmoid=True
                    )
                model.fit(
                    dataset_train=train_data,
                    dataset_valid=valid_data,
                    evaluation_step=10,
                    metric='logloss',
                    patience=200,
                    device=device,
                    reg_term=reg_type,
                )
                ce = model.test(test_data, 'logloss')
                csv_writer.writerow([ce, weight, reg_type, seed])
    res_file.close()

    update_dict=False
    if update_dict:
        # Make dictionary 
        hyper_param_dict = {}
        res_dict = defaultdict(list)
        weight_list = []
        with open('./hyper_param_results.csv','r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                ce = float(row[0])
                weight = row[1]
                reg_type = row[2]
                res_dict[(reg_type, weight)].append(ce)
                if weight not in weight_list:
                    weight_list.append(weight)

        for reg_type in reg_types:
            ce_means = [np.mean(res_dict[(reg_type, weight)]) for weight in weight_list]
            print(reg_type, ce_means)
            best_index = np.argmin(ce_means)
            hyper_param_dict[reg_type] = float(weight_list[best_index])
        hyper_param_dict['None'] = 0.0
        with open('reg_weight_dict.pkl', 'wb+')as f:
            pickle.dump(hyper_param_dict, f)




    