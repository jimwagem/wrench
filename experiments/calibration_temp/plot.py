import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    tau_type = 'tau2'
    result_path = f'./results_{tau_type}.csv'
    df = pd.read_csv(result_path)

    grouped = df.groupby('Temperature')
    means = grouped.mean()
    stds = grouped.std()
    ece = means['ECE'].to_numpy()
    acc = means['acc'].to_numpy()
    weight = means['weight'].to_numpy()
    ece_std = stds['ECE'].to_numpy()
    acc_std = stds['acc'].to_numpy()
    weight_std = stds['weight'].to_numpy()

    temp_index = means.index.to_numpy()
    index = np.arange(len(temp_index))

    # ECE plot
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.plot(index, ece, label='ece')
    ax.fill_between(index, ece + ece_std, ece - ece_std, alpha=0.3)
    ax.plot(index, acc, label='acc')
    ax.fill_between(index, acc + acc_std, acc - acc_std, alpha=0.3)
    ax.legend()
    ax.xaxis.set_ticks(index)
    ax.xaxis.set_ticklabels(temp_index)
    ax.set_ylim((0,1))
    ax.set_xlabel('Temperature')
    ax.set_title('Effect of temperature on calibration for WeaSEL')
    plt.savefig(f'./temp_cal_{tau_type}.png')
    plt.show()

    # Weight plot
    fig = plt.figure(1)
    ax = fig.add_subplot()
    one_index = 1
    assert temp_index[one_index] == 1.0
    lin_fit = temp_index * weight[one_index]
    ax.plot(index, weight, label='mean $\\tau_2 \\theta$')
    ax.fill_between(index, weight + weight_std, weight - weight_std, alpha=0.3)
    ax.plot(index, lin_fit, label='linear fit')
    ax.legend()
    ax.xaxis.set_ticks(index)
    ax.xaxis.set_ticklabels(temp_index)
    ax.set_xlabel('Temperature')
    ax.set_title('Effect of temperature on mean accuracy score for WeaSEL')
    plt.savefig(f'./temp_weight_{tau_type}.png')
    plt.show()
    plt.clf()