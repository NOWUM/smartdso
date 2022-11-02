import os
# -> steps and corresponding time resolution strings in pandas
RESOLUTION = {1440: "min", 96: "15min", 24: "h"}
# -> timescaledb connection to store the simulation results
DATABASE_URI = os.getenv(
    "DATABASE_URI", "postgresql://opendata:opendata@10.13.10.41:5432/smartdso"
)
SUB_GRID = int(os.getenv("SUB_GRID", 5))

KEY = 0


def key_generator(pv_capacity, ev_capacity):
    global KEY
    KEY += 1
    return f"S{SUB_GRID}C{KEY}_{pv_capacity}_{ev_capacity}"
