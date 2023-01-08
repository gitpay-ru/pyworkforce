import json
import os

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pprint import PrettyPrinter
from pyworkforce.utils.shift_spec import get_shift_coverage, get_12h_transitional_shifts, build_non_sequential_shifts, unwrap_shift
from pyworkforce.plotters.matplotlib import plot
import pandas as pd
import numpy as np
from pyworkforce.plotters.scheduling import plot, plot_xy_per_interval

scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(scheduler_data_path, '../scheduling_output.json'), 'r') as f:
    solution = json.load(f)

resources_shifts = solution['resources_shifts']
df = pd.DataFrame(resources_shifts)

required_resources = []
df0 = df[df['day'] == 0]

# print(df0)

s1 = "Day_9_12_45"
t1 = unwrap_shift(s1)
print(t1)

s2 = "Night_9_21_0"
t2 = unwrap_shift(s2)
print(t2)

t3 = np.array(t1) | np.array(t2)
print(t3)

previous = 0
count = 1
for c in t3:
    if previous == 0 and c == 0:
        count += 1
    previous = c

print(count)
print(count / 4.0)
print(count / 4.0  >= 12)


shifts = ["Day_9_6_13_15", "Night_9_21_22_15"]
shifts_coverage = get_shift_coverage(shifts, with_breaks=True)
shift_names = list(shifts_coverage.keys())
print(shift_names)
# t = get_12h_transitional_shifts(shift_names)
print("++++++++++++++")
t = build_non_sequential_shifts(shift_names, h_distance = 12, m_step = 15)
print(t)

with open('../non_sequential_shifts.json', 'w') as f:
    f.write(json.dumps(t, indent=2))