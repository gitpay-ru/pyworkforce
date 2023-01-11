from pathlib import Path
import json
import pandas as pd
import numpy as np
from pyworkforce.utils.shift_spec import required_positions, get_shift_short_name, get_shift_coverage
import math
from datetime import datetime as dt
from pyworkforce.utils.common import get_datetime

class MultiZonePlanner():
    def __init__(self,
        input_csv_path: str,
        input_meta_path: str,
        output_dir: str):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(input_csv_path, parse_dates=[0], index_col=0)

        with open(input_meta_path, 'r') as f:
            self.meta = json.load(f)

    def solve(self):
        print("Start")
        shift = self.meta['shifts'][0] #todo map
        shift_names = [get_shift_short_name(shift)]
        shifts_spec = get_shift_coverage(shift_names, with_breaks=True)
        # cover_check = [int(any(l)) for l in zip(*shifts_spec.values())]

        self.df['positions'] = self.df.apply(lambda row: required_positions(row['call_volume'], row['aht'], 15, row['art'], row['service_level']), axis=1)

        # Prepare required_resources
        HMin = 60
        DayH = 24
        min_date = min(self.df.index)
        max_date = max(self.df.index)
        days = (max_date - min_date).days + 1
        # print(self.df.index[1])
        date_diff = self.df.index[1] - self.df.index[0]
        step_min = int(date_diff.total_seconds() / HMin)
        ts = int(HMin / step_min)

        # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        self.df.index = self.df.index.tz_localize(tz='Europe/Moscow')
        print(self.df)

        # self.df = self.df.shift(periods=-3, fill_value=0)
        # print(self.df)
        # exit()
        
        # Employes partitions by timezone from meta
        manpowers = np.array(list(map(lambda t: float(t['dup']), self.meta['employees'])))
        print(manpowers)
        # Get ratios
        manpowers = manpowers / manpowers.sum(axis = 0)
        print(manpowers / manpowers.sum(axis = 0))
        # Get timezones
        timezones = list(map(lambda t: int(t['utc']), self.meta['employees'])) #todo GMT
        print(timezones)

        campaignUtc = int(self.meta['campaignUtc'])

        parties = list(zip(manpowers, timezones))
        for i in parties:
            tzone_shift = i[1] - campaignUtc
            df = self.df.copy()

            df['positions_quantile'] = df['positions'].apply(lambda t: math.ceil(t * i[0]))
            df = df.shift(periods=(-1 * ts * tzone_shift), fill_value = 0)

            required_resources = []
            for i in range(days):
                df_short = df[i * DayH * ts : (i + 1) * DayH * ts]
                required_resources.append(df_short['positions_quantile'].tolist())
                # print(required_resources)
                # exit()

            # todo:
            print("Do scheduling")
            print("Do rostering")
            print("combine")


        result = {
            "status": "TBD1",
            "cost": -1,
        }

        return result