from pathlib import Path
import json
import pandas as pd
import numpy as np
from pyworkforce.utils.shift_spec import required_positions

class MultiZonePlanner():
    def __init__(self,
        input_csv_path: str,
        input_meta_path: str,
        output_dir: str):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.csv = pd.read_csv(input_csv_path)

        with open(input_meta_path, 'r') as f:
            self.meta = json.load(f)

    def solve(self):
        print("Start")

        df = pd.read_csv('../scheduling_input.csv', parse_dates=[0], index_col=0)
        df['positions'] = df.apply(lambda row: required_positions(row['call_volume'], row['aht'], 15, row['art'], row['service_level']), axis=1)
        print(df)

        # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        df.index = df.index.tz_localize(tz='Europe/Moscow')
        
        # Guess employess partitions from meta
        manpowers = np.array(list(map(lambda t: t['dup'], self.meta['employees'])))
        # Get ratios
        manpowers = manpowers / manpowers.sum(axis = 0)
        # Get timezones
        timezones = list(map(lambda t: t['utc'], self.meta['employees'])) #todo GMT

        print(manpowers)
        print(manpowers / manpowers.sum(axis = 0))
        print(timezones)


        result = {
            "status": "TBD1",
            "cost": -1,
        }

        return result