import logging
import torch
from wrench.dataset import load_dataset, resplit_dataset
from wrench.logging import LoggingHandler
from wrench.classification import WeaSEL
from wrench.utils import set_seed
import pandas as pd
import numpy as np

if __name__=="__main__":
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logger = logging.getLogger(__name__)

    device = torch.device('cpu')

    #### Load dataset
    dataset_path = '../../../datasets/'
    data = 'youtube'
    # bert_model_name = 'bert-base-cased'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert',
        cache_name='bert',
        device=device,
    )

    tau_type = 'tau2'
    results_path=f'./results_{tau_type}.csv'
    columns=['Temperature', 'ECE', 'acc']
    
    results = []
    temp_range = [0.5, 1, 2, 4, 8, 16]
    # temp_range = [0.25, 0.5, 1, 1.5, 2, 3, 4]
    # temp_range = [0.1, 0.05, 5, 7, 10, 15]
    n_seeds = 10
    for temp in temp_range:
        for seed in range(n_seeds):
            if tau_type == 'tau1':
                tau1 = temp
                tau2_factor = 1.0
            elif tau_type == 'tau2':
                tau1 = 1.0
                tau2_factor = temp
            
            model = WeaSEL(
                temperature=tau1,
                acc_scaler_factor=tau2_factor,
                dropout=0.3,
                hidden_size=70,

                batch_size=64,
                real_batch_size=64,
                test_batch_size=64,
                n_steps=10000,
                grad_norm=1.0,

                backbone='MLP',
                # backbone='BERT',
                backbone_model_name='MLP',
                backbone_fine_tune_layers=-1,  # fine  tune all
                optimizer='AdamW',
                optimizer_lr=1e-4,
                optimizer_weight_decay=7e-7
            )
            model.fit(
                dataset_train=train_data,
                dataset_valid=valid_data,
                evaluation_step=10,
                metric='auc',
                patience=100,
                device=device
            )
            ece = model.test(test_data, 'ece')
            acc = model.test(test_data, 'acc')
            logger.info(f'WeaSEL test ece, acc for temp {temp} and seed {seed}: {ece}, {acc}')
            results.append([temp, ece, acc])
    
    df = pd.DataFrame(data=np.array(results), columns=columns)
    df.to_csv(path_or_buf=results_path, mode='a')
    



