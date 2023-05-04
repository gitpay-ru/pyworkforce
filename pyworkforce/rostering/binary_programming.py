import numpy as np
from ortools.sat.python import cp_model


# https://github.com/google/or-tools/blob/master/examples/python/shift_scheduling_sat.py
def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.
  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence

def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.
  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                # name = f': under_span({start}, {length})'
                name = ""
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                # name = f': over_span({start}, {length})'
                name = ""
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])

    return cost_literals, cost_coefficients

class MinHoursRoster:
    """

    It assigns a list of resources to a list of required positions per day and shifts; it takes into account
    different restrictions as shift bans, consecutive shifts, resting days, and others.
    It also introduces soft restrictions like shift preferences.
    The "optimal" criteria is defined as the minimum total scheduled hours,
    optionally weighted by resources shifts preferences

    Parameters
    ----------

    num_days: int,
        Number of days needed to schedule
    resources: list[str],
        Resources available to shift
    resources_min_w_hours: list[str],
        Min working hours for each employee
    resources_max_w_hours: list[str],
        Max working hours for each employee
    shifts: list,
        Array of shifts names
    shifts_hours: list,
        Array of size [shifts] with the total hours within the shift
    required_resources: dict[list]
        Each key of the dict must be one of the shifts, the value must be a  list of length [days]
        specifying the number of resources to shift in each day for that shift
    max_search_time: float, default = 240
        Maximum time in seconds to search for a solution
    num_search_workers: int, default = 2
        Number of workers to search for a solution
    shift_constraints: list
        Work days contraints
    rest_constraints: list
        Rest days contraints
    logging: boolean
    """

    def __init__(self, num_days: int,
                 resources: list,
                 resources_min_w_hours: list,
                 resources_max_w_hours: list,
                 shifts: list,
                 shifts_hours: list,
                 required_resources: list,
                 max_search_time: float = 540,
                 num_search_workers=2,
                 shift_constraints = [],
                 rest_constraints = [],
                 logging = False):

        self.num_days = num_days
        self.resources = resources
        self.resources_min_w_hours = resources_min_w_hours
        self.resources_max_w_hours = resources_max_w_hours
        self.num_resource = len(self.resources)
        self.shifts = shifts
        self.num_shifts = len(shifts)
        self.shifts_hours = shifts_hours
        self.required_resources = required_resources
        self.max_search_time = max_search_time
        self.num_search_workers = num_search_workers
        self.__deficit_weight = 1
        self.shift_constraints = shift_constraints
        self.rest_constraints = rest_constraints
        self.logging = logging

        self._status = None
        self.solver = None

    def solve(self):
        """
        Runs the optimization solver

        Returns
        -------

        solution : dict,
            Dictionary that contains the status on the optimization, the list of resources to shift in each day
            and the list of resources resting for each day
        """

        sch_model = cp_model.CpModel()

        # Decision Variable

        # shifted_resource: 1 if resource n is shifted for day d in shift s
        shifted_resource = np.empty(shape=(self.num_resource, self.num_days, self.num_shifts), dtype='object')
        for n in range(self.num_resource):
            for d in range(self.num_days):
                for s in range(self.num_shifts):
                    shifted_resource[n][d][s] = sch_model.NewBoolVar(f'resource_shifts_n{n}d{d}s{s}')

        # Constraints

        objective_int_vars = []
        objective_int_coeffs = []
        objective_bool_vars = []
        objective_bool_coeffs = []

        # The number of shifted resource must be >= that required resource, for each day and shift
        for d in range(self.num_days):
            for s in range(self.num_shifts):
                works = [shifted_resource[n][d][s] for n in range(self.num_resource)]
                delta = sch_model.NewIntVar(0, self.num_resource, f'delta_d{d}s{s}')
                sch_model.Add(sum(works) >= self.required_resources[self.shifts[s]][d] - delta)
                sch_model.Add(sum(works) <= self.required_resources[self.shifts[s]][d] + delta)

                objective_int_vars.append(delta)
                objective_int_coeffs.append(self.__deficit_weight)

        # A resource can at most, work 1 shift per day
        for n in range(self.num_resource):
            for d in range(self.num_days):
                sch_model.Add(sum(shifted_resource[n][d][s] for s in range(self.num_shifts)) <= 1)

        intA = 4 #aligner based on 15 mins slot
        # Min w h
        for n in range(self.num_resource):
            sch_model.Add(
                sum(shifted_resource[n][d][s] * int(intA * self.shifts_hours[s])
                    for d in range(self.num_days) for s in range(self.num_shifts)) >= int(intA * self.resources_min_w_hours[n]))
        
        # Max w h
        for n in range(self.num_resource):
            sch_model.Add(
                sum(shifted_resource[n][d][s] * int(intA * self.shifts_hours[s])
                    for d in range(self.num_days) for s in range(self.num_shifts)) <= int(intA * self.resources_max_w_hours[n]))

        # Resource shift constraints -- for all employees are same (per day)
        # 1. First we create a new matrix which represents resource working per day

        # daily_resource: 1 if resource is working on day d (sum(shifts) == 1)
        daily_resource = np.empty(shape=(self.num_resource, self.num_days), dtype='object')
        for n in range(self.num_resource):
            for d in range(self.num_days):
                # daily_resource[n][d] = sch_model.NewIntVar(0, 1, f'resource_day_n{n}d{d}')
                daily_resource[n][d] = sch_model.NewBoolVar(f'resource_day_n{n}d{d}')

        # 2. Apply constraint on daily work - no more than 1 shift per day
        for n in range(self.num_resource):
            for d in range(self.num_days):
                sch_model.Add(
                    sum(shifted_resource[n][d][s] for s in range(self.num_shifts)) == daily_resource[n][d]
                )

        # 3. Apply sequence constraints of resource daily work
        for ct in self.shift_constraints:
            (hard_min, soft_min, min_cost, soft_max, hard_max, max_cost) = ct
            for n in range(self.num_resource):
                works = [daily_resource[n,d] for d in range(self.num_days)]
                variables, coeffs = add_soft_sequence_constraint(
                    sch_model, works,
                    hard_min, soft_min, min_cost, soft_max, hard_max, max_cost,
                    f'resource_shifts_constraint_n{n}'
                )
                objective_bool_vars.extend(variables)
                objective_bool_coeffs.extend(coeffs)

        # 4. Apply sequence constraints of resource rests between workdays,
        for rct in self.rest_constraints:
            (hard_min, soft_min, min_cost, soft_max, hard_max, max_cost) = rct
            for n in range(self.num_resource):
                not_works = [daily_resource[n, d].Not() for d in range(self.num_days)]
                variables, coeffs = add_soft_sequence_constraint(
                    sch_model, not_works,
                    hard_min, soft_min, min_cost, soft_max, hard_max, max_cost,
                    f'resource_rest_constraint_n{n}'
                )
                objective_bool_vars.extend(variables)
                objective_bool_coeffs.extend(coeffs)

        # Objective function: Minimize the total number of shifted hours rewarded by resource preferences
        sch_model.Minimize(
            sum(objective_int_vars[i] * objective_int_coeffs[i] for i in range(len(objective_int_vars)))
            +
            sum(objective_bool_vars[i] * objective_bool_coeffs[i] for i in range(len(objective_bool_vars)))
        )

        print("Solving started...")
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.max_search_time
        self.solver.num_search_workers = self.num_search_workers
        self.solver.parameters.log_search_progress = self.logging

        solution_printer = cp_model.ObjectiveSolutionPrinter()
        self._status = self.solver.Solve(sch_model, solution_printer)

        # Output
        if self._status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            resource_shifts = []
            resting_resource = []
            shifted_hours = 0
            for n in range(self.num_resource):
                for d in range(self.num_days):
                    working = False
                    for s in range(self.num_shifts):
                        if self.solver.Value(shifted_resource[n][d][s]):
                            resource_shifts.append({
                                'id': n,
                                "resource": self.resources[n],
                                "day": d,
                                "shift": self.shifts[s]})
                            working = True
                            shifted_hours += self.shifts_hours[s]
                    if not working:
                        resting_resource.append({
                            "resource": self.resources[n],
                            "day": d
                        })

            solution = {"status": self.solver.StatusName(self._status),
                        "cost": self.solver.ObjectiveValue(),
                        "shifted_hours": shifted_hours,
                        "total_resources": len(self.resources),
                        "total_shifts": len(resource_shifts),
                        "resting_days": len(resting_resource),
                        "resource_shifts": resource_shifts,
                        "resting_resource": resting_resource}
        else:
            solution = {"status": self.solver.StatusName(self._status),
                        "cost": -1,
                        "shifted_hours": -1,
                        "total_resources": 0,
                        "total_shifts": 0,
                        "resting_days": 0,
                        "resource_shifts": [{'resource': -1, 'day': -1, 'shift': 'Unknown'}],
                        "resting_resource": [{'resource': -1, 'day': -1}]}

        return solution
