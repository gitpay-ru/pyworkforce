import json
import os

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pyworkforce.rostering.binary_programming import MinHoursRoster
from pprint import PrettyPrinter


from pyworkforce import plotters

scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(scheduler_data_path, '15mins_rostering_data.json'), 'r') as f:
    shifts_info = json.load(f)

pp = PrettyPrinter(indent=2)

solver = MinHoursRoster(num_days=shifts_info["num_days"],
                        resources=shifts_info["resources"],
                        shifts=shifts_info["shifts"],
                        shifts_hours=shifts_info["shifts_hours"],
                        min_working_hours=shifts_info["min_working_hours"],
                        max_resting=shifts_info["max_resting"],
                        non_sequential_shifts=shifts_info["non_sequential_shifts"],
                        banned_shifts=shifts_info["banned_shifts"],
                        required_resources=shifts_info["required_resources"],
                        resources_preferences=shifts_info["resources_preferences"],
                        resources_prioritization=shifts_info["resources_prioritization"])

solution = solver.solve()
# pp.pprint(solution)
print(solution)


shifts_spec =    {"Morning9": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 "Morning12": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 "Afternoon": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     "Night": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                     "Mixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

shift_colors = {
  "Morning9":'#7EA7D8',
  'Morning12': '#FFF79A',
  'Afternoon': '#A1D372',
  'Night':'#EB4845',
  'Mixed':'#7BCDC8'
}

plotters.matplotlib.plot(solution, shifts_spec, shifts_info["num_days"], shift_colors, fig_size=(12,5))
