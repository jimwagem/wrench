import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    result_path = './results.csv'
    df = pd.read_csv(result_path)

    grouped = df.groupby('Temperature')
    means = grouped.mean()
    stds = grouped.std()
    ece = means['ECE'].to_numpy()
    acc = means['acc'].to_numpy()
    ece_std = stds['ECE'].to_numpy()
    acc_std = stds['acc'].to_numpy()

    temp_index = means.index.to_numpy()
    index = np.arange(len(temp_index))

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
    plt.savefig('./temp_cal.png')
    plt.show()