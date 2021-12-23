import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__=="__main__":
    # files = ['./results/multi_gauss_snorkel.csv', './results/multi_gauss_flyingsquid.csv', './results/multi_gauss_weasel.csv']
    # Read accuracy files
    names=['snorkel', 'flyingsquid', 'weasel']
    lf_types = ['bad', 'random', 'constant']
    acc_files = [f'./results/multi_gauss_{name}.csv' for name in names]
    
    what_to_plot='weights'

    if what_to_plot == 'acc':
        bad_results = []
        random_results = []
        constant_results = []
        n_good_lf = 5
        for f in acc_files:
            bad_file_results = []
            random_file_results = []
            constant_file_results = []
            idxs = set()
            with open(f, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    t=row[0]
                    idx = int(row[1])
                    mean=float(row[2])
                    std=float(row[3])
                    if t=='bad':
                        bad_file_results.append((mean, std))
                    elif t=='random':
                        random_file_results.append((mean, std))
                    elif t=='constant':
                        constant_file_results.append((mean, std))
                    idxs.add(idx)
                bad_results.append(bad_file_results)
                random_results.append(random_file_results)
                constant_results.append(constant_file_results)

        # Make plot
        x_range = list(sorted(idxs))
        fig, axs = plt.subplots(3, 1, sharex=True)
        for name, results in zip(names, bad_results):
            means = np.array([e[0] for e in results])
            stds = np.array([e[1] for e in results])
            axs[0].plot(x_range, means, label=name)
            axs[0].fill_between(x_range, means+stds, means-stds, alpha=0.3)
            axs[0].set_ylim([0, 1.05])

        for name, results in zip(names, random_results):
            means = np.array([e[0] for e in results])
            stds = np.array([e[1] for e in results])
            axs[1].plot(x_range, means, label=name)
            axs[1].fill_between(x_range, means+stds, means-stds, label=name, alpha=0.3)
            axs[1].set_ylim([0, 1.05])
        for name, results in zip(names, constant_results):
            means = np.array([e[0] for e in results])
            stds = np.array([e[1] for e in results])
            axs[2].plot(x_range, means, label=name)
            axs[2].fill_between(x_range, means+stds, means-stds, label=name, alpha=0.3)
            axs[2].set_ylim([0, 1.05])
            axs[2].set_xticks(x_range)

            # Plot ratio of good lfs on top
        top = axs[0].twiny()
        top.set_xticks(axs[2].get_xticks())
        top.set_xticklabels([np.round(n_good_lf / (n_good_lf + v), 2) for v in axs[2].get_xticks()])
        top.set_xlabel('Ratio of good labeling functions')

        axs[0].legend()
        axs[1].set_ylabel('Accuracy')
        axs[2].set_xlabel('Number of adversarial labeling functions')

        axs[0].set_title('Orthogonal task')
        axs[1].set_title('Random')
        axs[2].set_title('Constant')
        fig.suptitle('Effect of adversarial labeling function on classification')
        plt.tight_layout()
        plt.savefig("./results/multi_gaussian.png")
        plt.show()
    
    elif what_to_plot=='weights':
        # Read weights files
        weight_files = [f'./results/multi_gauss_weights_{name}.csv' for name in names]
        dfs = [pd.read_csv(weight_file) for weight_file in weight_files]
        df_means = [df.groupby(['lf_type','n_lfs']).mean() for df in dfs]
        df_stds = [df.groupby(['lf_type','n_lfs']).std() for df in dfs] 
        # Plot the weights
        fig, axs = plt.subplots( len(names), len(lf_types), sharex=True)
        # for i, name in enumerate(names):
        #     mean_df = df_means[i]
        #     index = mean_df.loc['random'].index.to_numpy()
        #     random_good_means = mean_df.loc['random']['good_mean'].to_numpy()
        #     random_bad_means = mean_df.loc['random']['bad_mean'].to_numpy()
        #     constant_good_means = mean_df.loc['constant']['good_mean'].to_numpy()
        #     constant_bad_means = mean_df.loc['constant']['bad_mean'].to_numpy()
        #     bad_good_means = mean_df.loc['bad']['good_mean'].to_numpy()
        #     bad_bad_means = mean_df.loc['bad']['bad_mean'].to_numpy()

        #     axs[0,0].plot(index, bad_bad_means, label=name)
        #     axs[0,1].plot(index, bad_good_means, label=name)
        #     axs[1,0].plot(index, random_bad_means, label=name)
        #     axs[1,1].plot(index, random_good_means, label=name)
        #     axs[2,0].plot(index, constant_bad_means, label=name)
        #     axs[2,1].plot(index, constant_good_means, label=name)
        #     axs[0,0].set_xticks(index)
        
        for i, lf_type in enumerate(lf_types):
            axs[0,i].set_xlabel(lf_type)
            for j, name in enumerate(names):
                mean_df = df_means[j]
                std_df = df_stds[j]
                index = mean_df.loc[lf_type].index.to_numpy()
                good_means = mean_df.loc[lf_type]['good_mean'].to_numpy()
                bad_means = mean_df.loc[lf_type]['bad_mean'].to_numpy()
                good_stds = std_df.loc[lf_type]['good_mean'].to_numpy()
                bad_stds = std_df.loc[lf_type]['bad_mean'].to_numpy()

                axs[j,i].plot(index, good_means, label='good')
                axs[j,i].fill_between(index, good_means + good_stds, good_means - good_stds, alpha=0.3)
                axs[j,i].plot(index, bad_means, label='bad')
                axs[j,i].fill_between(index, bad_means + bad_stds, bad_means - bad_stds, alpha=0.3)
                if i == 0:
                    axs[j,i].set_ylabel(name)
                elif i == 2:
                    axs[j,i].set_xticks(index)
                
        
        axs[0,0].legend()
        # axs[0,0].set_title('Orthogonal task good')
        # axs[1,0].set_title('Random good')
        # axs[2,0].set_title('Constant good')
        # axs[0,1].set_title('Orthogonal task bad')
        # axs[1,1].set_title('Random bad')
        # axs[2,1].set_title('Constant bad')

        fig.suptitle('Average labeling function weight')
        plt.show()




