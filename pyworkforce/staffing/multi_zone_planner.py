from pathlib import Path
import json
import pandas as pd
import numpy as np
from pyworkforce.utils.shift_spec import required_positions, get_shift_short_name, get_shift_coverage, unwrap_shift
from pyworkforce.plotters.scheduling import plot_xy_per_interval
import math
from datetime import datetime as dt
from pyworkforce.utils.common import get_datetime
from pyworkforce.scheduling import MinAbsDifference
from pyworkforce.rostering.binary_programming import MinHoursRoster

class MultiZonePlanner():
    def __init__(self,
        input_csv_path: str,
        input_meta_path: str,
        output_dir: str):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(input_csv_path, parse_dates=[0], index_col=0)

        with open(input_meta_path, 'r') as f:
            self.meta = json.load(f)

        # Set timezones
        self.timezones = list(map(lambda t: int(t['utc']), self.meta['employees']))

    def dump_stat_and_plot(self, tzone, solution, df):
        resources_shifts = solution['resources_shifts']
        df3 = pd.DataFrame(resources_shifts)
        df3['shifted_resources_per_slot'] = df3.apply(lambda t: np.array(unwrap_shift(t['shift'])) * t['resources'], axis=1)
        df4 = df3[['day', 'shifted_resources_per_slot']].groupby('day', as_index=False)['shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis = 0)).to_frame()
        np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
        df4.to_csv(f'../shifted_resources_per_slot_{tzone}.csv')
        arr = df4['shifted_resources_per_slot'].values
        arr = np.concatenate(arr)
        df['resources_shifts'] = arr.tolist()
        df.to_csv(f'../scheduling_output_stage2_{tzone}.csv')

        plot_xy_per_interval(f'{tzone}.png', df, x='index', y=["positions", "resources_shifts"])

    def dump_scheduling_output_rostering_input(self, tzone, days, num_resources, solution, shifts_spec):
        with open(f'../scheduling_output_{tzone}.json', 'w') as f:
                f.write(json.dumps(solution, indent=2))

        resources_shifts = solution['resources_shifts']
        df1 = pd.DataFrame(resources_shifts)
        df2 = df1.pivot(index='shift', columns='day', values='resources').rename_axis(None, axis=0)

        df2['combined']= df2.values.tolist()

        rostering = {}
        rostering['num_days'] = days
        rostering['num_resources'] = num_resources
        rostering['shifts'] = list(shifts_spec.keys())
        rostering['min_working_hours'] = 176 # Dec 2022 #todo:
        rostering['max_resting'] = 9 # Dec 2022
        rostering['non_sequential_shifts'] = []
        rostering['required_resources'] = df2['combined'].to_dict()
        rostering['banned_shifts'] = []
        rostering['resources_preferences'] = []
        rostering['resources_prioritization'] = []

        with open(f'../scheduling_output_rostering_input_{tzone}.json', 'w') as outfile:
            outfile.write(json.dumps(rostering, indent=2))

    def schedule(self):
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

        date_diff = self.df.index[1] - self.df.index[0]
        step_min = int(date_diff.total_seconds() / HMin)
        ts = int(HMin / step_min)

        # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        self.df.index = self.df.index.tz_localize(tz='Europe/Moscow')
        
        # Employes partitions by timezone from meta
        manpowers = np.array(list(map(lambda t: int(t['dup']), self.meta['employees'])))

        # Get ratios
        manpowers_r = manpowers / manpowers.sum(axis = 0)
        print(manpowers, manpowers_r)

        campaignUtc = int(self.meta['campaignUtc'])

        parties = list(zip(manpowers_r, self.timezones, manpowers))
       
        for party in parties:
            tzone_shift = party[1] - campaignUtc
            df = self.df.copy()

            df['positions_quantile'] = df['positions'].apply(lambda t: math.ceil(t * party[0]))
            df = df.shift(periods=(-1 * ts * tzone_shift), fill_value = 0)

            required_resources = []
            for i in range(days):
                df_short = df[i * DayH * ts : (i + 1) * DayH * ts]
                required_resources.append(df_short['positions_quantile'].tolist())
                # print(required_resources)
                # exit()

            scheduler = MinAbsDifference(num_days = days,  # S
                                periods = DayH * ts,  # P
                                shifts_coverage = shifts_spec,
                                required_resources = required_resources,
                                max_period_concurrency = int(df['positions_quantile'].max()),  # gamma
                                max_shift_concurrency=int(df['positions_quantile'].mean()),  # beta
                                )
            solution = scheduler.solve()

            tzone = str(party[1])

            self.dump_scheduling_output_rostering_input(
                tzone,
                days,
                int(party[2]), #num_resources == manpowers
                solution,
                shifts_spec
            )

            self.dump_stat_and_plot(
                tzone,
                solution,
                df.copy()
            )

        result = {
            "status": "Done",
            "cost": -1,
        }

        return result

    def roster(self):
        for i in self.timezones:
            print(f'Timezone: {i}')
            with open(f'../scheduling_output_rostering_input_{i}.json', 'r') as f:
                shifts_info = json.load(f)

            shift_names = shifts_info["shifts"]
            shifts_hours = [int(i.split('_')[1]) for i in shifts_info["shifts"]]
            print(shift_names, shifts_hours)

            RATIO = 1.5
            resources = [f'emp_{i}' for i in range(0, int(RATIO * shifts_info["num_resources"]) )]

            solver = MinHoursRoster(num_days=shifts_info["num_days"],
                        resources=resources,
                        shifts=shifts_info["shifts"],
                        shifts_hours=shifts_hours,
                        min_working_hours=shifts_info["min_working_hours"],
                        max_resting=shifts_info["max_resting"],
                        non_sequential_shifts=shifts_info["non_sequential_shifts"],
                        banned_shifts=shifts_info["banned_shifts"],
                        required_resources=shifts_info["required_resources"],
                        resources_preferences=shifts_info["resources_preferences"],
                        resources_prioritization=shifts_info["resources_prioritization"])

            solution = solver.solve()

            with open(f'../rostering_output_{i}.json', 'w') as f:
                f.write(json.dumps(solution, indent = 2))

        return "TBD"