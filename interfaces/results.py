from sqlalchemy import create_engine, inspect
import pandas as pd

engine = create_engine('postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
tables = inspect(engine).get_table_names()


def get_scenarios():
    if len(tables) > 0:
        scenarios = dict()
        for table in tables:
            query = f'Select Distinct scenario from {table}'
            scenarios[table] = set([value._data[0] for value in engine.execute(query).fetchall()])
        return set.intersection(*scenarios.values())
    else:
        return set()


def get_iterations(scenario):
    if len(tables) > 0:
        iteration = dict()
        for table in tables:
            query = f"Select Distinct iteration from {table} where scenario='{scenario}'"
            iteration[table] = set([value._data[0] for value in engine.execute(query).fetchall()])
        return set.intersection(*iteration.values())
    else:
        return set()


def get_car_usage(scenario: str, iteration: str):
    query = f"Select time, odometer, soc, work, errand, hobby from cars where scenario='{scenario}' " \
            f"and iteration='{iteration}'"
    dataframe = pd.read_sql(query, engine).set_index('time')
    dataframe.index = pd.to_datetime(dataframe.index)
    car_usage = dataframe.sort_index()

    query = f"Select avg(evs) as evs, avg(distance) as distance, " \
            f"avg(demand) as demand from evs where scenario='{scenario}' group by time"
    dataframe = pd.read_sql(query, engine)
    dataframe.index = pd.to_datetime(dataframe.index)
    total_ev = dataframe.sort_index()

    return car_usage, total_ev


def get_simulation_results(type_: str, scenario: str, iteration: str, aggregate: str = 'avg'):
    if iteration == 'total':
        query = f"Select time, {aggregate}({type_}) as {type_} from meta where scenario='{scenario}'" \
                f" group by time order by time"
    else:
        query = f"Select time, {type_} from meta where scenario='{scenario}' and iteration={iteration} " \
                f"order by time"
    dataframe = pd.read_sql(query, engine).set_index('time')
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe = dataframe.resample('5min').mean()
    dataframe[type_] = dataframe[type_].apply(lambda x: round(x, 2))
    # print(dataframe.head(5))
    return dataframe


def get_transformers(scenario: str, sub_id: str):
    # -> get the five with the highest utilization
    if sub_id != 'total':
        query = f"select transformers.id_, time, utilization as util from transformers " \
                f"join (select distinct id_, max(utilization) as util from transformers where scenario='{scenario}' " \
                f"and grid='{sub_id}'" \
                f"group by id_ order by util desc limit 5) " \
                f"as first5 on first5.id_=transformers.id_ and first5.util=transformers.utilization"
    else:
        query = f"select transformers.id_, time, utilization as util from transformers " \
                f"join (select distinct id_, max(utilization) as util from transformers where scenario='{scenario}' " \
                f"group by id_ order by util desc limit 5) " \
                f"as first5 on first5.id_=transformers.id_ and first5.util=transformers.utilization"

    max_utilization = pd.read_sql(query, engine).set_index('time')
    return max_utilization


def get_lines(scenario: str, sub_id: str):
    # -> get the five with the highest utilization
    if sub_id != 'total':
        query = f"select lines.id_, time, utilization as util from lines " \
                f"join (select distinct id_, max(utilization) as util from lines where scenario='{scenario}' " \
                f"and grid='{sub_id}'" \
                f"group by id_ order by util desc limit 5) " \
                f"as first5 on first5.id_=lines.id_ and first5.util=lines.utilization"
    else:
        query = f"select lines.id_, time, utilization as util from lines " \
                f"join (select distinct id_, max(utilization) as util from lines where scenario='{scenario}' " \
                f"group by id_ order by util desc limit 5) " \
                f"as first5 on first5.id_=lines.id_ and first5.util=lines.utilization"

    max_utilization = pd.read_sql(query, engine).set_index('time')

    if sub_id != 'total':
        query = f"Select id_, max(utilization) as utilization from lines where scenario='{scenario}' and grid='{sub_id}' " \
                f"group by id_"
    else:
        query = f"Select id_, max(utilization) as utilization from lines where scenario='{scenario}' group by id_"

    total_utilization = pd.read_sql(query, engine).set_index('id_')

    return max_utilization, total_utilization


def get_transformer_utilization(scenario:str, sub_id: str):
    query = f"Select time, avg(utilization) as util from transformers where scenario='{scenario}' " \
            f"and grid='{sub_id}' group by time order by time"

    return pd.read_sql(query, engine).set_index('time')


if __name__ == "__main__":
    min_shifted = get_simulation_results(type_='shifted', scenario='EV100LIMIT-1', iteration='total', aggregate='min')
    max_shifted = get_simulation_results(type_='shifted', scenario='EV100LIMIT-1', iteration='total', aggregate='max')
    avg_shifted = get_simulation_results(type_='shifted', scenario='EV100LIMIT-1', iteration='total', aggregate='avg')
    shifted = pd.DataFrame(data=dict(min=min_shifted['shifted'].values, max=max_shifted['shifted'].values,
                                     avg=avg_shifted['shifted'].values), index=avg_shifted.index)

    # car_use, evs = get_car_usage(scenario='high', iteration='high')