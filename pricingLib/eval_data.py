import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    charging_data = np.load(r'./pricingLib/charging_data.pkl')
    sizes = [50, 100, 1000, 2500, 5000, 10000]
    # sizes = [2500, 10000]
    p_size = 100
    for size in tqdm(sizes):
        mean = []
        for _ in tqdm(range(10_000)):
            data = charging_data[np.random.choice(range(size), p_size), :].mean(axis=0)
            mean.append(data)

        mean = np.asarray(mean)
        m = mean.mean(axis=0)
        plt.plot(m)
    plt.legend(sizes)
    #
    # for k in range(10):
    #     plt.plot(mean[k, :])
    #
    # plt.plot(m, 'r--')
    #
    # mean = []
    # for _ in range(1000):
    #     data = charging_data[np.random.choice(range(100), 50), :].mean(axis=0)
    #     mean.append(data)
    #
    # mean = np.asarray(mean)
    # m = mean.mean(axis=0)
    #
    # plt.plot(m, 'b--')

    plt.show()
