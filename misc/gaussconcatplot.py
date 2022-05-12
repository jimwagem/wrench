import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
if __name__=="__main__":
    n = 500
    xu = np.concatenate((np.random.normal(0, 1, size=n),np.random.normal(10, 1, size=n)))
    xl = np.concatenate((np.random.normal(0, 1, size=n),np.random.normal(10, 1, size=n)))
    np.random.shuffle(xu)
    np.random.shuffle(xl)
    df = pd.DataFrame({'xu':xu, 'xl':xl})
    sns.jointplot(data=df, x="xu", y="xl",kind='kde')
    plt.savefig('./gaussconcat.pdf')
    plt.show()