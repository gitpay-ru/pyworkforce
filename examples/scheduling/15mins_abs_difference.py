"""
Requirement: Find the number of workers needed to schedule per shift in a production plant for the next 2 days with the
    following conditions:
    * There is a number of required persons per hour and day given in the matrix "required_resources"
    * There are 4 available scheduling called "Morning", "Afternoon", "Night", "Mixed"; their start and end hour is
      determined in the dictionary "shifts_coverage", 1 meaning the shift is active at that hour, 0 otherwise
    * The number of required workers per day and period (hour) is determined in the matrix "required_resources"
    * The maximum number of workers that can be shifted simultaneously at any hour is 25, due plat capacity restrictions
    * The maximum number of workers that can be shifted in a same shift, is 20
"""
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pyworkforce.scheduling import MinAbsDifference
from pyworkforce.queuing import ErlangC
from pprint import PrettyPrinter
import pandas as pd
import math
import json
import numpy as np
from collections import deque

from pyworkforce.plotters.scheduling import plot, plot_xy_per_interval
from pyworkforce.utils.shift_spec import get_shift_coverage, get_shift_colors, decode_shift_spec, required_positions, unwrap_shift
from pyworkforce.utils.common import get_datetime


MAX_POS = int(5.0 / 7 * 375)
MAX_PERIOD_CONCURRENCY = int(5.0 / 7 * 375)

df = pd.read_csv('../scheduling_input.csv')

# def required_positions(call_volume, aht, interval, art, service_level):
#   erlang = ErlangC(transactions=call_volume, aht=aht / 60.0, interval=interval, asa=art / 60.0, shrinkage=0.0)
#   positions_requirements = erlang.required_positions(service_level=service_level / 100.0, max_occupancy=1.00)
#   return positions_requirements['positions']

df['positions'] = df.apply(lambda row: required_positions(row['call_volume'], row['aht'], 15, row['art'], row['service_level']), axis=1)
# df['cut_positions'] = df.apply(lambda row: row['positions'] if row['positions'] <= MAX_POS else MAX_POS, axis=1)
df.to_csv('../scheduling_output_stage1.csv')

min_date = get_datetime(min(df['tc']))
max_date = get_datetime(max(df['tc']))
days = (max_date - min_date).days + 1
date_diff = get_datetime(df.iloc[1]['tc']) - get_datetime(df.iloc[0]['tc'])
step_min = int(date_diff.total_seconds() / 60)
HMin = 60
DayH = 24
ts = int(HMin / step_min)
required_resources = []
for i in range(days):
  df0 = df[i * DayH * ts : (i + 1) * DayH * ts]
  # required_resources.append(df0['cut_positions'].tolist())
  required_resources.append(df0['positions'].tolist())

# shifts = ["Day_9_6_13_15"]#, "Night_9_21_22_15"]
# shifts = ["Night_9_21_22_15"]

# MSK
# shifts1 = ["Day_9_6_13_15"]
# shifts2 = ["Day_9_10_16_15"] 
# shifts = shift1 + shift2 <- csv_orig




shifts_spec = get_shift_coverage(shifts, with_breaks=True)

cover_check = [int(any(l)) for l in zip(*shifts_spec.values())]
print(cover_check)
# exit()

scheduler = MinAbsDifference(num_days = days,  # S
                                 periods = 24 * ts,  # P
                                 shifts_coverage = shifts_spec,
                                 required_resources = required_resources,
                                #  max_period_concurrency = MAX_PERIOD_CONCURRENCY,
                                #  max_shift_concurrency = MAX_POS,
                                 max_period_concurrency = int(df['positions'].max()),  # gamma
                                 max_shift_concurrency=int(df['positions'].mean()),  # beta
                                 )

solution = scheduler.solve()
# pp = PrettyPrinter(indent=2)
# pp.pprint(solution)

with open('../scheduling_output.json', 'w') as outfile:
    outfile.write(json.dumps(solution, indent=2))

shift_names = list(shifts_spec.keys())
shift_colors = get_shift_colors(shift_names)

resources_shifts = solution['resources_shifts']

df1 = pd.DataFrame(resources_shifts)
df2 = df1.pivot(index='shift', columns='day', values='resources').rename_axis(None, axis=0)

df2['combined']= df2.values.tolist()

rostering = {}
rostering['num_days'] = days
rostering['num_resources'] = 375
rostering['shifts'] = list(shifts_spec.keys())
rostering['min_working_hours'] = 176 # Dec 2022
rostering['max_resting'] = 9 # Dec 2022
rostering['non_sequential_shifts'] = []
rostering['required_resources'] = df2['combined'].to_dict()
rostering['banned_shifts'] = []
rostering['resources_preferences'] = []
rostering['resources_prioritization'] = []

with open('../scheduling_output_rostering_input.json', 'w') as outfile:
    outfile.write(json.dumps(rostering, indent=2))

# Stat
resources_shifts = solution['resources_shifts']
df3 = pd.DataFrame(resources_shifts)
df3['shifted_resources_per_slot'] = df3.apply(lambda t: np.array(unwrap_shift(t['shift'])) * t['resources'], axis=1)
df4 = df3[['day', 'shifted_resources_per_slot']].groupby('day', as_index=False)['shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis = 0)).to_frame()
np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
df4.to_csv('../shifted_resources_per_slot.csv')
arr = df4['shifted_resources_per_slot'].values
arr = np.concatenate(arr)
# df3 = pd.read_csv('../scheduling_output_stage1.csv')
df['resources_shifts'] = arr.tolist()
df.to_csv('../scheduling_output_stage2.csv')

plot_xy_per_interval("scheduling_Night_9_21_22_15.png", df, x="tc", y=["positions", "resources_shifts"])
