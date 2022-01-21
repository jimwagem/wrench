import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# def plot_sum_weights(file_name):
#     df = pd.read_csv(file_name)
#     # Unused columns
#     df = df.drop(columns=['n_good_lfs', 'acc', 'mcc', 'run_id'])
#     groups = df.groupby(['model_name', 'lf_type', 'n_lfs'])
#     df_agg = groups.agg(['mean', 'std'])
#     weasel_bad = df_agg.loc['weasel','bad']

#     index = weasel_bad.index.to_numpy()
#     good_mean = weasel_bad['good_mean','mean'].to_numpy()
#     bad_mean = weasel_bad['bad_mean','mean'].to_numpy()

#     plt.plot(index, good_mean*5, label='good_scaled')
#     plt.plot(index, bad_mean*index, label='bad_scaled')
#     plt.plot(index, bad_mean, label='bad')
#     plt.gca().set_xticklabels(index)
#     plt.gca().set_xticks(list(range(len(index))))
#     plt.legend()
#     plt.show()
    


def plot_weights(file_name, show=True, models=None):
    # Number of standard deviations
    n_sigma = 2
    # n_sigma = 3

    df = pd.read_csv(file_name)

    # Unused columns
    df = df.drop(columns=['n_good_lfs', 'acc', 'mcc', 'run_id'])

    groups = df.groupby(['model_name', 'lf_type', 'n_lfs'])
    names = models or df['model_name'].unique()
    lf_types = df['lf_type'].unique()

    df_agg = groups.agg(['mean', 'std'])
    
    names = {
        'bad': "Orthogonal task",
        'random': 'Random',
        'constant': 'Constant',
    }

    # Plot the weights
    fig, axs = plt.subplots(len(names), len(lf_types), sharex=True, figsize=(10, 6))
    for i, lf_type in enumerate(lf_types):
        axs[0, i].set_xlabel(lf_type)
        for j, name in enumerate(names):
            res = df_agg.loc[name, lf_type]
            index = res.index.to_numpy()
            good_means = res['good_mean']['mean']
            good_stds = res['good_mean']['std'] * n_sigma
            bad_means = res['bad_mean']['mean']
            bad_stds = res['bad_mean']['std'] * n_sigma
            # total_means = res['total_mean']['mean']
            # total_std = res['total_std']['mean'] * n_sigma

            x_range = list(range(len(index)))
            axs[j, i].plot(x_range, good_means, label='bad')
            axs[j, i].fill_between(x_range, good_means + good_stds, good_means - good_stds, alpha=0.3)
            axs[j, i].plot(x_range, bad_means, label='orthogonal')
            axs[j, i].fill_between(x_range, bad_means + bad_stds, bad_means - bad_stds, alpha=0.3)
            # axs[j, i].plot(x_range, total_means, label='total')
            # axs[j, i].fill_between(x_range, total_means + total_std, total_means - total_std, alpha=0.3)
            axs[j, i].set_xticks(x_range)
            if i == 0:
                axs[j, i].set_ylabel(names[name])
            elif i == 2:
                axs[j, i].set_xticklabels(index)

    axs[0, 0].legend()

    fig.suptitle(f'Average labeling function weight\nCI = ${n_sigma}\sigma$')
    plt.tight_layout()
    models_suffix = ''
    if models is not None:
        models_suffix += "_"
        models_suffix += "_".join(models)
    plt.savefig(f"./results/multi_gaussian_weights_std{n_sigma}{models_suffix}.pdf")
    if show:
        plt.show()


def plot_metric(file_name, show=True, models=None):
    results = pd.read_csv(file_name)

    # Number of standard deviations for uncertainty
    n_sigma = 2  # 95%
    # n_sigma = 3  # 99%

    metrics = {
        'acc': 'Accuracy',
        'mcc': 'Matthews Correlation Coefficient'
    }
    n_lf = results['n_lfs'].unique()
    tasks = results['lf_type'].unique()
    n_good_lf = results['n_good_lfs'].unique()[0]
    run_counts = results.groupby('run_id')['model_name'].count()
    n_runs_complete = len(run_counts[run_counts == max(run_counts)])
    n_runs_total = len(run_counts)

    names = {
        'bad': "Orthogonal task",
        'random': 'Random',
        'constant': 'Constant',
    }

    # Make plot
    for metric, metric_name in metrics.items():
        x_labels = list(sorted(n_lf))
        x_range = list(range(len(n_lf)))
        fig, axs = plt.subplots(len(tasks), 1, sharex=True, figsize=(10, 8))
        for task_id, task_name in enumerate(tasks):
            task_results = results[results['lf_type'] == task_name]
            for name, group in task_results.groupby('model_name'):
                # Skip if models are selected
                if models is not None and name not in models:
                    continue
                res = group.groupby('n_lfs')[metric].agg(['mean', 'std'])
                means = res['mean'].values
                stds = res['std'].values * n_sigma
                axs[task_id].plot(
                    x_range,
                    means,
                    label=name
                )
                axs[task_id].fill_between(
                    x_range,
                    means + stds,
                    means - stds,
                    alpha=0.3
                )
                axs[task_id].set_ylim([0, 1.05])
                axs[task_id].set_title(names[task_name])
                axs[task_id].set_xticks(x_range)

        # Plot ratio of good lfs on top
        top = axs[0].twiny()
        top.set_xticks(axs[2].get_xticks())
        top.set_xticklabels([np.round(n_good_lf / (n_good_lf + v), 2) for v in x_labels])
        top.set_xlabel('Ratio of good labeling functions')
        axs[2].set_xticklabels(x_labels)

        axs[0].legend()
        axs[1].set_ylabel(metric_name)
        axs[2].set_xlabel('Number of adversarial labeling functions')

        if n_runs_complete < n_runs_total:
            runs_text = f'{n_runs_complete} runs complete, {n_runs_total} total'
        else:
            runs_text = f'{n_runs_complete} runs'
        fig.suptitle(f'Effect of adversarial labeling function on classification\nCI = ${n_sigma}\sigma$ $\quad$ {runs_text}')
        plt.tight_layout()

        models_suffix = ''
        if models is not None:
            models_suffix += "_"
            models_suffix += "_".join(models)
        plt.savefig(f"./results/multi_gaussian_{metric}{models_suffix}_std{n_sigma}.pdf")
        if show:
            plt.show()


if __name__ == "__main__":
    file_name = "./results/multi_gauss_new.csv"
    # show = False
    # plot_metric(file_name, show=show, models=['snorkel', 'flyingsquid', 'weasel'])
    # plot_metric(file_name, show=show)
    # plot_weights(file_name, show=show, models=['snorkel', 'flyingsquid', 'weasel'])
    # plot_weights(file_name, show=show)
    plot_sum_weights(file_name)
