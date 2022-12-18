import json
import os

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pyworkforce.rostering.binary_programming import MinHoursRoster
from pprint import PrettyPrinter
from pyworkforce.utils.shift_spec import get_shift_coverage

from pyworkforce.plotters.matplotlib import plot

from pyworkforce import plotters

scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(scheduler_data_path, '../rostering_output.json'), 'r') as f:
    solution = json.load(f)

pp = PrettyPrinter(indent=2)
# pp.pprint(solution)
# exit()




t = solution['resource_shifts']

t_m = list(filter(lambda x: x['day'] == 0 and x['shift'] == 'Day_9_6_0', t))

# print(t_m)
# print(f'Day_9_6_0: {len(t_m)}')
# exit()

# t_m = list(filter(lambda x: x['day'] == 1 and x['shift'] == 'Morning', t1))
# t_a = list(filter(lambda x: x['day'] == 1 and x['shift'] == 'Afternoon', t1))
# t_n = list(filter(lambda x: x['day'] == 1 and x['shift'] == 'Night', t1))
# t_mix = list(filter(lambda x: x['day'] == 1 and x['shift'] == 'Mixed', t1))
# print(f'Morning: {len(t_m)}')
# print(f'Afternoon: {len(t_a)}')
# print(f'Night: {len(t_n)}')
# print(f't_mix: {len(t_mix)}')
# shift_colors = {
#   "Morning":'#34eb46',
#   'Afternoon': '#f55142',
#   # 'Morning12_6_30': '#A1D372',
#   # 'Morning12_6_45': '#A1D372',
#   'Night':'#0800ff',
#   'Mixed':'#ffff00',
#   # 'Night12_18_30':'#EB4845',
#   # 'Night12_18_45':'#7BCDC8'
# }

# with open(os.path.join(scheduler_data_path, 'final_shift_spec.json'), 'r') as f:
#     shifts_spec = json.load(f)

shifts = ["Day_9_6_13_15", "Night_9_21_22_15"]
shifts_spec = get_shift_coverage(shifts)


shift_colors = {}
for i in shifts_spec.keys():
  if "Day" in i:
    shift_colors[i] = '#34eb46'
  else:
    shift_colors[i] = '#0800ff'


plot(solution, shifts_spec, 15, 31, shift_colors, "res.png", fig_size=(15,8))
