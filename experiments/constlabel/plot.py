import matplotlib.pyplot as plt
import csv
import numpy as np


if __name__=='__main__':
    res_file = './results_1000_patience.csv'
    # res_file = './results.csv'
    f = open(res_file, 'r')

    csv_reader = csv.reader(f)

    acc_dict = dict()
    zero_frac_dict = dict()
    for row in csv_reader:
        ap = float(row[0])
        acc = float(row[1])
        zero_frac = float(row[2])
        if not ap in acc_dict:
            acc_dict[ap] = []
        if not ap in zero_frac_dict:
            zero_frac_dict[ap] = []
        acc_dict[ap].append(acc)
        zero_frac_dict[ap].append(zero_frac)
    
    mode = 'scatter'
    if mode == 'mean_std':
        # Number of std multiples
        n = 2
        index = list(acc_dict.keys())
        print(acc_dict[ap] for ap in index)
        zero_frac_mean = np.array([np.mean(zero_frac_dict[ap]) for ap in index])
        zero_frac_std = np.array([np.std(zero_frac_dict[ap]) for ap in index])
        acc_mean = np.array([np.mean(acc_dict[ap]) for ap in index])
        acc_std = np.array([np.std(acc_dict[ap]) for ap in index])


        # index = [float(ap) for ap in index]
        plt.plot(index, zero_frac_mean, label='zero fraction')
        plt.fill_between(index, zero_frac_mean + n*zero_frac_std, zero_frac_mean - n*zero_frac_std, alpha=0.3)
        plt.plot(index, acc_mean, label='accuracy')
        plt.fill_between(index, acc_mean + n*acc_std, acc_mean - n*acc_std, alpha=0.3)
        plt.legend()
        plt.xlabel('Abstaining probability')
        plt.title('Effect of constant zero label on WeaSEL')
        plt.show()
    elif mode == 'scatter':
        zero_frac_index = []
        zero_frac_val = []
        for k, l in zero_frac_dict.items():
            for val in l:
                zero_frac_index.append(k)
                zero_frac_val.append(val)
        acc_index = []
        acc_val = []
        for k, l in acc_dict.items():
            for val in l:
                acc_index.append(k)
                acc_val.append(val)
        index = list(acc_dict.keys())
        zero_frac_mean = np.array([np.mean(zero_frac_dict[ap]) for ap in index])
        acc_mean = np.array([np.mean(acc_dict[ap]) for ap in index])
        
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].scatter(zero_frac_index, zero_frac_val, label='zero fraction', c='blue')
        axs[0].set_ylabel('Zero fraction')
        axs[0].plot(index, zero_frac_mean)
        axs[0].set_ylim((0.0, 1.1))
        axs[1].scatter(acc_index, acc_val, label='accuracy', c='orange')
        axs[1].set_ylabel('Accuracy')
        axs[1].plot(index, acc_mean, c='orange')
        # axs[1].set_ylim((0.0, 1.1))
        axs[1].set_xlabel('Abstaining probability')
        fig.suptitle('Effect of constant zero label on WeaSEL')
        plt.savefig('./plot.png')
        plt.show()
