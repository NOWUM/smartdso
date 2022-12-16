from pricingLib.energy_provider import optimize
from multiprocessing import Pool


if __name__ == "__main__":
    sizes = [10, 20, 30, 50, 100, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 25_000]

    with Pool(12) as worker:
        worker.map(optimize, sizes)