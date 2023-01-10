import json
import os

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pprint import PrettyPrinter
from pyworkforce.utils.shift_spec import get_shift_coverage, unwrap_shift
from pyworkforce.plotters.matplotlib import plot
import pandas as pd
import numpy as np
from pyworkforce.plotters.scheduling import plot_xy_per_interval
# from random import randrange, randint
import random

scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(scheduler_data_path, '../rostering_output.json'), 'r') as f:
    solution = json.load(f)

total_resources = solution['total_resources']
NUM_EMPLOYEE = 375

id_to_drop = random.sample(range(0, total_resources), total_resources - NUM_EMPLOYEE)

resource_shifts = solution['resource_shifts']
ddf = pd.DataFrame(resource_shifts)

ddf = ddf[~ddf['id'].isin(id_to_drop)]
print(ddf)
ddf.to_csv('test.csv')

df = ddf
# df = ddf.head(8272)
# df.to_csv('test1.csv')
# exit()


df['shifted_resources_per_slot'] = df.apply(lambda t: np.array(unwrap_shift(t['shift'])) * 1, axis=1)

df1 = df[['day', 'shifted_resources_per_slot']].groupby('day', as_index=False)['shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis = 0)).to_frame()
# print(df1)
# exit()
np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
# df1.to_csv('test.csv')
arr = df1['shifted_resources_per_slot'].values
arr = np.concatenate(arr)

df3 = pd.read_csv('../scheduling_output_stage1.csv')
df3['resources_shifts'] = arr.tolist()
# print(df3)
# exit()

plot_xy_per_interval("rostering2.png", df3, x="tc", y=["positions", "resources_shifts"])

# df1.to_csv('test.csv')

# shifts = ["Day_9_6_13_15", "Night_9_21_22_15"]
# shifts_spec = get_shift_coverage(shifts)

# shift_colors = {}
# for i in shifts_spec.keys():
#   if "Day" in i:
#     shift_colors[i] = '#34eb46'
#   else:
#     shift_colors[i] = '#0800ff'

# plot(solution, shifts_spec, 15, 31, shift_colors, "../Dec2022_562_employees.png", fig_size=(15,8))
