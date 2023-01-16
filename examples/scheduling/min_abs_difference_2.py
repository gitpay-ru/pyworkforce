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

from pyworkforce.scheduling.shifts_selection_2 import MinAbsDifference2

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pyworkforce.scheduling import MinAbsDifference
from pprint import PrettyPrinter

# Columns are an hour of the day, rows are the days
# N_dp
required_resources = [
                              [9, 11, 17, 9, 7, 12, 5, 11, 8, 9, 18, 17, 8, 12, 16, 8, 7, 12, 11, 10, 13, 19, 16, 7],
    # [13, 13, 12, 15, 18, 20, 13, 16, 17, 8, 13, 11, 6, 19, 11, 20, 19, 17, 10, 13, 14, 23, 16, 8]
]

# Each entry of a shift, is an hour of the day (24 columns)
# E_sp
shifts_coverage = {"Morning": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 "Afternoon": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     "Night": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                     "Mixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

# (shift, concurrency)
shifts_concurency = [
    # max 7 people at morning
    (0, 10),
    (1, 10),
    (2, 10),
    (3, 10)
]

scheduler = MinAbsDifference2(num_days=1,  # S
                              periods=24,  # P
                              shifts_coverage=shifts_coverage,
                              required_resources=required_resources,
                              max_periods_concurrency=[(p, 27) for p in range(24)],  # gamma
                              max_shifts_concurrency=shifts_concurency)   # beta

solution = scheduler.solve()
pp = PrettyPrinter(indent=2)
pp.pprint(solution)
