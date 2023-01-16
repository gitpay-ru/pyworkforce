import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from pyworkforce.scheduling.base import BaseShiftScheduler
from pyworkforce.scheduling.utils import check_positive_integer, check_positive_float


class MinAbsDifference2(BaseShiftScheduler):
    def __init__(self, num_days: int,
                 periods: int,
                 shifts_coverage: dict,
                 required_resources: list,
                 max_periods_concurrency: int,
                 max_shifts_concurrency: list,
                 max_search_time: float = 120.0,
                 num_search_workers=2,
                 *args, **kwargs):
        """
        The "optimal" criteria is defined as the number of resources per shift
        that minimize the total absolute difference between the required resources
        per period and the actual scheduling found by the solver

        Parameters
        ----------

        num_days: int,
            Number of days needed to schedule
        periods: int,
            Number of working periods in a day
        shifts_coverage: dict,
            dict with structure {"shift_name": "shift_array"} where "shift_array" is an array of size [periods] (p), 1 if shift covers period p, 0 otherwise
        required_resources: list,
            Array of size [days, periods]
        max_period_concurrency: int,
            Maximum resources that are allowed to shift in any period and day
        max_shifts_concurrency: int,
            Number of maximum allowed resources in the same shift
        max_search_time: float, default = 240
            Maximum time in seconds to search for a solution
        num_search_workers: int, default = 2
            Number of workers to search for a solution
        """

        is_valid_num_days = check_positive_integer("num_days", num_days)
        is_valid_periods = check_positive_integer("periods", periods)
        for (p, c) in max_periods_concurrency:
            is_valid_max_period_concurrency = check_positive_integer(f"max_period_concurrency_{p}", c)
        for (s, c) in max_shifts_concurrency:
            is_valid_max_shift_concurrency = check_positive_integer(f"max_shifts_concurrency_{s}", c)
        is_valid_max_search_time = check_positive_float("max_search_time", max_search_time)
        is_valid_num_search_workers = check_positive_integer("num_search_workers", num_search_workers)

        if periods != len(max_periods_concurrency):
            raise ValueError(f"Shift coverage and concurrency dimensions are not equal")

        if len(shifts_coverage.keys()) != len(max_shifts_concurrency):
            raise ValueError(f"Shift coverage and concurrency dimensions are not equal")

        self.num_days = num_days
        self.shifts = list(shifts_coverage.keys())
        self.num_shifts = len(self.shifts)
        self.num_periods = periods
        self.shifts_coverage_matrix = list(shifts_coverage.values())
        self.max_shifts_concurrency = max_shifts_concurrency
        self.max_periods_concurrency = max_periods_concurrency
        self.required_resources = required_resources
        self.max_search_time = max_search_time
        self.num_search_workers = num_search_workers
        self.solver = cp_model.CpSolver()
        self.transposed_shifts_coverage = None
        self.status = None

    def solve(self):
        """
        Runs the optimization solver

        Returns
        -------
        solution: dict,
            Dictionary with the status on the optimization, the resources to schedule per day and the
            final value of the cost function
        """
        sch_model = cp_model.CpModel()

        # Resources: Number of resources assigned in day d to shift s
        resources = np.empty(shape=(self.num_days, self.num_shifts), dtype='object')
        # transition resources: Variable to change domain coordinates from min |x-a|
        # to min t, s.t t>= x-a and t>= a-x
        transition_resources = np.empty(shape=(self.num_days, self.num_periods), dtype='object')

        # Resources
        for d in range(self.num_days):
            for (s,c) in self.max_shifts_concurrency:
                resources[d][s] = sch_model.NewIntVar(0, c, f'resources_d{d}s{s}')

        for d in range(self.num_days):
            for (p, c) in self.max_periods_concurrency:
                transition_resources[d][p] = sch_model.NewIntVar(-c, c, f'transition_resources_d{d}p{p}')

        # Constrains

        # transition must be between x-a and a-x
        for d in range(self.num_days):
            for p in range(self.num_periods):
                sch_model.Add(transition_resources[d][p] >= (
                        sum(resources[d][s] * self.shifts_coverage_matrix[s][p] for s in range(self.num_shifts)) -
                        self.required_resources[d][p]))
                sch_model.Add(transition_resources[d][p] >= (self.required_resources[d][p]
                                                             - sum(resources[d][s] * self.shifts_coverage_matrix[s][p]
                                                                   for s in range(self.num_shifts))))

        # Total programmed resources, must be less or equals to max_period_concurrency, for each day and period
        for d in range(self.num_days):
            for (p, c) in self.max_periods_concurrency:
                sch_model.Add(
                    sum(resources[d][s] * self.shifts_coverage_matrix[s][p] for s in range(self.num_shifts)) <= c)

        # Objective Function: Minimize the absolute value of the difference between required and shifted resources

        sch_model.Minimize(
            sum(transition_resources[d][p] for d in range(self.num_days) for p in range(self.num_periods)))

        self.solver.parameters.max_time_in_seconds = self.max_search_time
        self.solver.num_search_workers = self.num_search_workers

        self.status = self.solver.Solve(sch_model)

        if self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            resources_shifts = []
            for d in range(self.num_days):
                for s in range(self.num_shifts):
                    resources_shifts.append({
                        "day": d,
                        "shift": self.shifts[s],
                        "resources": self.solver.Value(resources[d][s])})

            solution = {"status": self.solver.StatusName(self.status),
                        "cost": self.solver.ObjectiveValue(),
                        "resources_shifts": resources_shifts}
        else:
            solution = {"status": self.solver.StatusName(self.status),
                        "cost": -1,
                        "resources_shifts": [{'day': -1, 'shift': 'Unknown', 'resources': -1}]}

        return solution

