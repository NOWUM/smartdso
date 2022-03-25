import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from agents.flexibility_provider import FlexibilityProvider

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-12-31'))

input_set = {'employee_ratio': os.getenv('EMPLOYEE_RATIO', 0.7),
             'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),
             'minimum_soc': int(os.getenv('MINIMUM_SOC', 50)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 50))/100}

FlexProvider = FlexibilityProvider(**input_set)
participants = FlexProvider.participants

len_ = 1440 * ((end_date - start_date).days + 1)
time_range = pd.date_range(start=start_date, periods=len_, freq='min')
result = {key: pd.Series(data=np.zeros(len_), index=range(len_))
          for key in ['commits', 'rejects', 'requests', 'waiting', 'charged', 'shift',
                      'soc', 'price', 'ref_distance', 'ref_soc']}

if __name__ == "__main__":
    indexer = 0
    for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
        for d_time in pd.date_range(start=day, periods=1440, freq='min'):
            requests = FlexProvider.get_requests(d_time)
            for id_, request in requests.items():
                participants[id_].commit_charging(0, d_time)
                for node_id, parameters in request.items():
                    for power, duration in parameters:
                        result['charged'][indexer:indexer + duration] += power
            for participant in participants.values():
                participant.do(d_time)
            indexer += 1
