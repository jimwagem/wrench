import torch
import numpy as np
from wrench.classification import WeaSEL
from wrench.synthetic import MultiGaussian

if __name__=='__main__':
    device = torch.device('cpu')
    seeds = 20

    seed_results=[]
    for _ in range(seeds):
        mg = MultiGaussian(n_features=8, n_visible_features=7, n_good_lfs=5, n_bad_lfs=10, n_random_lfs=0, n_class=5, sample_low=-5, sample_high=5)
        train_dataset = mg.generate_split(split='train', n_data=10000)
        valid_dataset = mg.generate_split(split='valid', n_data=1000)
        test_dataset = mg.generate_split(split='test', n_data=1000)
        model = WeaSEL(
            temperature=1.0,
            dropout=0.3,
            hidden_size=100,

            batch_size=16,
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
        model.fit(
            dataset_train=train_dataset,
            dataset_valid=valid_dataset,
            evaluation_step=10,
            metric='acc',
            patience=100,
            device=device
            )
        seed_results.append(model.test(test_dataset, 'acc'))
    print(np.mean(seed_results))
    print(np.std(seed_results))