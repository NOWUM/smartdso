from multiprocessing import Pool
from single_car import sim_one_car
import numpy as np


if __name__ == "__main__":
    try:
        with Pool(12) as worker:
            r = worker.map(sim_one_car, np.arange(1, 12_000))
            matrix = np.asarray(r)
            np.save(open('charging.npy', 'wb'), matrix)
    except Exception as e:
        print(e)
        # pass

