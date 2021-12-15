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

def write_results(log_file, lf_type, acc, std, n_lfs):
    with open(log_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([lf_type, n_lfs, acc, std])

if __name__=='__main__':
    device = torch.device('cpu')
    seeds = 20

    # for n_lfs in range(0, 10):
    #     print(f'testing {n_lfs}')
    #     seed_results=[]
    #     for i in range(seeds):
    #         set_seed(i)
    #         mg = MultiGaussian(n_features=8, n_visible_features=7, n_good_lfs=5, n_bad_lfs=n_lfs, n_random_lfs=0, n_class=5, sample_low=-5, sample_high=5)
    #         train_dataset = mg.generate_split(split='train', n_data=10000)
    #         train_dataset.labels = None
    #         valid_dataset = mg.generate_split(split='valid', n_data=1000)
    #         test_dataset = mg.generate_split(split='test', n_data=1000)
    #         # model = WeaSEL(
    #         #     temperature=1.0,
    #         #     dropout=0.3,
    #         #     hidden_size=100,

    #         #     batch_size=64,
    #         #     real_batch_size=8,
    #         #     test_batch_size=128,
    #         #     n_steps=10000,
    #         #     grad_norm=1.0,

    #         #     backbone='MLP',
    #         #     # backbone='BERT',
    #         #     optimizer='AdamW',
    #         #     optimizer_lr=5e-5,
    #         #     optimizer_weight_decay=0.0,
    #         # )
    #         # model = ModelWrapper(
    #         #     model_func=lambda : EndClassifierModel(
    #         #         batch_size=128,
    #         #         test_batch_size=512,
    #         #         n_steps=1000,
    #         #         backbone='MLP',
    #         #         optimizer='Adam',
    #         #         optimizer_lr=1e-2,
    #         #         optimizer_weight_decay=0.0,
    #         #     ),
    #         #     label_model_func=lambda: FlyingSquid(),
    #         #     name = '2stage_MLP_flyingsquid'
    #         # )
    #         model = ModelWrapper(
    #             model_func=lambda : EndClassifierModel(
    #                 batch_size=128,
    #                 test_batch_size=512,
    #                 n_steps=1000,
    #                 backbone='MLP',
    #                 optimizer='Adam',
    #                 optimizer_lr=1e-2,
    #                 optimizer_weight_decay=0.0,
    #             ),
    #             label_model_func=lambda: Snorkel(
    #                 lr=0.01,
    #                 l2=0.0,
    #                 n_epochs=10
    #             ),
    #             name = '2stage_MLP_snorkel'
    #         )
    #         # model.fit(
    #         #     train_dataset,
    #         #     valid_dataset,
    #         #     10,
    #         #     metric='acc',
    #         #     patience=100,
    #         #     device=device,
    #         #     verbose=False
    #         #     )
    #         model.fit(
    #             train_dataset,
    #             valid_dataset,
    #             evaluation_step=10,
    #             metric='acc',
    #             patience=100,
    #             device=device,
    #             verbose=False
    #             )
    #         seed_results.append(model.test(test_dataset, 'acc'))
    #     print(np.mean(seed_results))
    #     print(np.std(seed_results))
    #     write_results('./results/multi_gauss_snorkel.csv', 'bad', np.mean(seed_results), np.std(seed_results), n_lfs)
    
    files = ['./results/multi_gauss_snorkel.csv', './results/multi_gauss_flyingsquid.csv', './results/multi_gauss_weasel.csv']
    names=['snorkel', 'flyingsquid', 'weasel']
    bad_results = []
    random_results = []
    x_range = np.arange(10)
    for f in files:
        bad_file_results = []
        random_file_results = []
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                t=row[0]
                mean=float(row[2])
                std=float(row[3])
                if t=='bad':
                    bad_file_results.append((mean, std))
                elif t=='random':
                    random_file_results.append((mean, std))
            bad_results.append(bad_file_results)
            random_results.append(random_file_results)
    
    fig, axs = plt.subplots(2, 1, sharex=True)
    for name, results in zip(names, bad_results):
        means = np.array([e[0] for e in results])
        stds = np.array([e[1] for e in results])
        axs[0].plot(x_range, means, label=name)
        axs[0].fill_between(x_range, means+std, means-std, alpha=0.3)
    for name, results in zip(names, random_results):
        means = np.array([e[0] for e in results])
        stds = np.array([e[1] for e in results])
        axs[1].plot(x_range, means, label=name)
        axs[1].fill_between(x_range, means+std, means-std, label=name, alpha=0.3)
    axs[0].legend()
    axs[0].set_ylabel('accuracy')
    axs[0].set_title('bad')
    axs[1].set_xlabel('Number of adverserial labeling function.')
    axs[1].set_title('random')
    fig.suptitle('Effect of adverserial labeling function on classification')
    plt.show()

