from ortools.sat.python.cp_model import CpSolver

from pyworkforce.breaks.breaks_intervals_scheduling_sat import BreaksIntervalsScheduling

INTERVALS_PER_HOUR = 4
class BreaksPrinter:
    def __init__(self,
                 model: BreaksIntervalsScheduling,
                 sat_solver: CpSolver,
                 sat_solver_status: any,
                 employee_calendar: dict,
                 breaks: list,
                 break_delays,
                 *args, **kwargs):

        self.num_intervals_per_day = INTERVALS_PER_HOUR * 24;
        self.num_days = int(num_intervals / self.num_intervals_per_day)

        self.num_intervals = num_intervals
        self.num_employees = num_employees
        self.employee_calendar = employee_calendar
        self.breaks = breaks
        (self.break_delays_min, self.break_delays_max) = break_delays

        self.intervals_demand = intervals_demand