import enum
import itertools
import math
import operator
from collections import defaultdict

from ortools.sat.python import cp_model

from pyworkforce.breaks.breaks_coverage import calculate_coverage
from pyworkforce.objective_solution_printer_with_limit import ObjectiveSolutionPrinterWithLimit
from pyworkforce.solver_params import SolverParams


class AdjustmentMode(enum.Enum):

    NoAdjustments = 1
    ByExpectedAverage = 2


INTERVALS_PER_HOUR = 4

class BreaksIntervalsScheduling:
    def __init__(self,
                 # intervals_demand: list, # not used for the moment
                 employee_calendar: dict,
                 breaks: list,
                 break_min_delay: int,
                 break_max_delay: int,
                 make_adjustments: AdjustmentMode = AdjustmentMode.NoAdjustments,
                 num_days: int = 31, # TODO: remove days, should calculate on the fly
                 solver_params: SolverParams = SolverParams.default(),
                 *args, **kwargs):

        # 'emp' -> [(shift_day_num, work_start_interval, work_end_interval)]
        self.employee_calendar = employee_calendar
        self.breaks = breaks
        self.break_min_delay = break_min_delay
        self.break_max_delay = break_max_delay
        self.adjustment_mode = make_adjustments
        self.num_days = num_days
        self.num_employees = len(self.employee_calendar)

        self.break_optimums = self.get_expected_breaks_coverage()

        self.__solver_params = solver_params

        # not using it now
        # self.intervals_demand = intervals_demand

        self.solver = None
        self.status = None

    def get_expected_breaks_coverage(self):

        breaks_duration = sum(
            [break_duration for (*_, break_duration) in self.breaks]
        )

        first_employee_shifts = next(iter(self.employee_calendar.values()))
        (_, start, end) = first_employee_shifts[0]
        shift_duration_per_day = end - start

        employee_coverage = calculate_coverage(self.employee_calendar, self.num_days*24*INTERVALS_PER_HOUR)

        optimums = list(map(
            lambda x: x*breaks_duration/shift_duration_per_day,
            employee_coverage
        ))

        return optimums

    def add_overlap_excess_penalty(self, model: cp_model.CpModel, employee_rest_intervals):

        assert len(employee_rest_intervals) == len(self.employee_calendar)

        num_intervals = len(self.break_optimums)
        num_employees = len(employee_rest_intervals)
        num_breaks = len(self.breaks)

        _FALSE = model.NewConstant(False)

        # overlaps [e, b, i] -> employees x breaks x intervals
        # overlaps = {}
        overlaps = defaultdict(lambda: _FALSE)

        # TODO: optimize intervals - don't compare unmatchable by design intervals

        for (ei, ekey) in enumerate(self.employee_calendar):

            # the idea is to match breaks only by corresponding shifts by days to minimize iterations,
            # because breaks from Tue can't overlap with breaks on Mon

            # for (day_num, work_start_int, work_end_int) in self.employee_calendar[ekey]:
            #     pass
            #
            # for (day_num, break_id, break_start_var, break_end_var) in employee_rest_intervals[ekey]:
            #     pass

            # rest_by_day_it = itertools.groupby(employee_rest_intervals[ekey], operator.itemgetter(0))  # 0 = day_num
            emp_rests_by_day = {}
            for (day_num, break_id, break_start_var, break_end_var) in employee_rest_intervals[ekey]:
                emp_rests_by_day.setdefault(day_num, []).append((break_start_var, break_end_var))

            for (day, work_start, work_end) in self.employee_calendar[ekey]:
                for bi, (*_, start, end) in enumerate(emp_rests_by_day[day]):
                    # Create overlap matrix only for intersected intervals!
                    for i in range(work_start, work_end):
                        overlap_i = model.NewBoolVar(f'overlap_e{ei}_b{bi}_i{i}')
                        before_i = model.NewBoolVar(f'before_e{ei}_b{bi}_i{i}')
                        after_i = model.NewBoolVar(f'after_e{ei}_b{bi}_i{i}')

                        model.Add(start <= i).OnlyEnforceIf(overlap_i)
                        model.Add(end > i).OnlyEnforceIf(overlap_i)  # Intervals are open ended on the right

                        model.Add(end <= i).OnlyEnforceIf(before_i)
                        model.Add(start > i).OnlyEnforceIf(after_i)

                        model.Add(overlap_i + before_i + after_i == 1)
                        overlaps[ei, bi, i] = overlap_i
                        # l. perron solution - end

        # for (e, breaks) in employee_rest_intervals.items():
        #     for b, br in enumerate(breaks):
        #         (*_, start, end) = br
        #
        #         # Create overlap matrix
        #         for i in range(num_intervals):
        #             overlap_i = model.NewBoolVar(f'overlap_e{e}_b{b}_i{i}')
        #             before_i = model.NewBoolVar(f'before_e{e}_b{b}_i{i}')
        #             after_i = model.NewBoolVar(f'after_e{e}_b{b}_i{i}')
        #
        #             model.Add(start <= i).OnlyEnforceIf(overlap_i)
        #             model.Add(end > i).OnlyEnforceIf(overlap_i)  # Intervals are open ended on the right
        #
        #             model.Add(end <= i).OnlyEnforceIf(before_i)
        #             model.Add(start > i).OnlyEnforceIf(after_i)
        #
        #             model.Add(overlap_i + before_i + after_i == 1)
        #             overlaps[e, b, i] = overlap_i
        #             # l. perron solution - end

        obj_vars = []

        for i in range(num_intervals):

            if self.break_optimums[i] > 0:
                epsilon = model.NewIntVar(0, num_employees, f'delta_i{i}')

                rest = [overlaps[e, b, i] for e in range(num_employees) for b in range(num_breaks)]
                model.Add(sum(rest) >= int(self.break_optimums[i]) - epsilon)
                model.Add(sum(rest) <= int(self.break_optimums[i]) + epsilon)

                obj_vars.append(epsilon)

        return obj_vars

    def solve(self):

        """Solves the shift scheduling problem."""

        print("Model building...")

        model = cp_model.CpModel()

        # 1. Remember employees' rest/breaks intervals
        # Will use it to get results per employee
        employee_rest = {}

        for e in self.employee_calendar:
            breaks = []

            for (shift_day_num, day_work_start_interval, day_work_end_interval) in self.employee_calendar[e]:

                # Accumulate intervals per working shift, order is not meaningful, hosts a bag of intervals
                # It will guarantee no intervals are intersected
                working_day = []

                for (i, br) in enumerate(self.breaks):
                    (_id, break_start_min, break_start_max, break_duration) = br
                    # breaks are relative to shift startings -> fix them:
                    break_start_min += day_work_start_interval
                    break_start_max += day_work_start_interval

                    # Break of given duration
                    start = model.NewIntVar(break_start_min, break_start_max, f'break_start_e{e}_br{i}')
                    duration = break_duration  # Python cp/sat code accepts integer variables or constants.
                    end = model.NewIntVar(day_work_start_interval, day_work_end_interval, f'break_end_e{e}_br{i}')
                    rest = model.NewIntervalVar(start, duration, end, f'break_e{e}_{i}')

                    # Working time after break, to be ensure they aren't joint
                    working_duration = model.NewIntVar(self.break_min_delay, self.break_max_delay, '')
                    working_end = model.NewIntVar(day_work_start_interval, day_work_end_interval + self.break_max_delay, '')
                    working = model.NewIntervalVar(end, working_duration, working_end, '')

                    breaks.append(
                        (shift_day_num, _id, start, end)
                    )

                    working_day.extend(
                        [rest, working]
                    )

                # This will require no overlaps of intervals during the shift.
                # Hence, will have a form ideally (w - work interval, n - break interval):
                #
                # employee calendar : ------------------■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■------------------
                # assigned breaks   : ------------------wwwwbwwwwwwwbbwwwwwwwwwwwwbwwwwwwwww------------------
                model.AddNoOverlap(working_day)

            # Remember what breaks were assigned to worker per shift
            employee_rest[e] = breaks

        penalties = []
        if self.adjustment_mode == AdjustmentMode.ByExpectedAverage:
            # 2. Minimize number of concurrent breaks
            penalties = self.add_overlap_excess_penalty(model, employee_rest)
            model.Minimize(sum(penalties))

        print("Solving started...")
        self.solver = cp_model.CpSolver()
        if self.__solver_params.max_iteration_search_time:
            self.solver.parameters.max_time_in_seconds = self.__solver_params.max_iteration_search_time
        if self.__solver_params.num_search_workers:
            self.solver.num_search_workers = self.__solver_params.num_search_workers
        if self.__solver_params.do_logging:
            self.solver.parameters.log_search_progress = self.__solver_params.do_logging
        if self.__solver_params.solution_limit:
            solution_printer = ObjectiveSolutionPrinterWithLimit(self.__solver_params.solution_limit)
        else:
            solution_printer = cp_model.ObjectiveSolutionPrinter()

        self.status = self.solver.Solve(model, solution_printer)

        # Output

        solution = {}
        if self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

            if self.__solver_params.do_logging:
                for p in penalties:
                    print(f'{p.Name()}: {self.solver.Value(p)}')

            scheduled_breaks = {}

            for e in self.employee_calendar.keys():
                emp_breaks = []
                for (shift_day_num, break_id, break_start, break_end) in employee_rest[e]:
                    start_index = self.solver.Value(break_start)
                    end_index = self.solver.Value(break_end)

                    emp_breaks.append(
                        (shift_day_num, break_id, start_index, end_index)
                    )

                scheduled_breaks[e] = emp_breaks

            solution = {
                "status": self.solver.StatusName(self.status),
                "cost": self.solver.ObjectiveValue(),
                "num_branches": self.solver.NumBranches(),
                "num_conflicts": self.solver.NumConflicts(),
                "wall_time": self.solver.WallTime(),
                "num_resources": len(self.employee_calendar),
                "resource_break_intervals": scheduled_breaks,
            }

        else:
            solution = {
                "status": self.solver.StatusName(self.status),
                "cost": -1,
                "num_branches": -1,
                "num_conflicts": -1,
                "wall_time": -1,
                "num_resources": len(self.employee_calendar.keys()),
                "resource_break_intervals": {},
            }

        return solution
