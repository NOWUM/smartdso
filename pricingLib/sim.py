from multiprocessing import Pool
from simple import sim_one_car as sim_simple_car
from t import sim_one_car as sim_optimize_car
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig()
logging.getLogger().setLevel('ERROR')

result = defaultdict(list)

CREATE = 'optimized'
# CREATE = 'simple'

if __name__ == "__main__":
    try:
        with Pool(52) as worker:
            if CREATE == 'simple':
                r = worker.map(sim_simple_car, np.arange(1, 12_000))
                matrix = np.asarray(r)
                np.save(open('charging.npy', 'wb'), matrix)
            elif CREATE == 'optimized':
                r = worker.map(sim_optimize_car, np.arange(1, 12_000))
                for hpfc, charging in r:
                    result[hpfc].append(charging)

    except Exception as e:
        print(e)
        # pass

