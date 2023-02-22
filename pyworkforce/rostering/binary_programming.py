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
    shifts: list,
        Array of shifts names
    shifts_hours: list,
        Array of size [shifts] with the total hours within the shift
    min_working_hours: int,
        Minimum working hours per resource in the horizon
    banned_shifts: list[dict]
        Each element {"resource": resource_index, "shift": shift_name, "day": day_number} indicating
        that the resource can't be assigned to that shift that particular day
        example: banned_shifts": [{"resource":"e.johnston@randatmail.com", "shift": "Night", "day":  0}],
    max_resting: int,
        Maximum number of resting days per resource in the total interval
    required_resources: dict[list]
        Each key of the dict must be one of the shifts, the value must be a  list of length [days]
        specifying the number of resources to shift in each day for that shift
    non_sequential_shifts: List[dict]
        Each element must have the form {"origin": first_shift, "destination": second_shift}
        to make sure that destination shift can't be after origin shift.
        example: [{"origin":"Night", "destination":"Morning"}]
    resources_preferences: list[dict]
        Each element must have the form {"resource": resource_idx, "shifts":shift_name}
        indicating the resources that have preference for shift
    resources_prioritization: list[dict], default=None
        Each element must have the form {"resource": resource_idx, "weight": weight_percentage}
        this represent the relative importance for resources_preferences assignment
    max_search_time: float, default = 240
        Maximum time in seconds to search for a solution
    num_search_workers: int, default = 2
        Number of workers to search for a solution
    """

    def __init__(self, num_days: int,
                 resources: list,
                 shifts: list,
                 shifts_hours: list,
                 min_working_hours: int,
                 banned_shifts: list,
                 max_resting: int,
                 required_resources: list,
                 non_sequential_shifts: list = None,
                 resources_preferences: list = None,
                 resources_prioritization: list = None,
                 max_search_time: float = 540,
                 num_search_workers=2,
                 strict_mode = True,
                 shift_constraints = [],
                 max_shifts_count: int = 0):

        self._num_days = num_days
        self.resources = resources
        self.num_resource = len(self.resources)
        self.shifts = shifts
        self.num_shifts = len(shifts)
        self.shifts_hours = shifts_hours
        self.min_working_hours = min_working_hours
        self.max_shifts_count = max_shifts_count
        self.banned_shifts = banned_shifts
        self.max_resting = max_resting
        self.required_resources = required_resources
        self.non_sequential_shifts = non_sequential_shifts
        self.resources_preferences = resources_preferences
        self.resources_prioritization = resources_prioritization
        self.max_search_time = max_search_time
        self.num_search_workers = num_search_workers
        self.non_sequential_shifts_indices = None
        self.resources_shifts_preferences = None
        self.resources_shifts_weight = None

        self.strict_mode = strict_mode
        self.__deficit_weight = 1
        self.shift_constraints = shift_constraints

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
        shifted_resource = np.empty(shape=(self.num_resource, self._num_days, self.num_shifts), dtype='object')
        for n in range(self.num_resource):
            for d in range(self._num_days):
                for s in range(self.num_shifts):
                    shifted_resource[n][d][s] = sch_model.NewBoolVar(f'resource_shifts_n{n}d{d}s{s}')
        # Constraints

        objective_int_vars = []
        objective_int_coeffs = []
        objective_bool_vars = []
        objective_bool_coeffs = []

        # The number of shifted resource must be >= that required resource, for each day and shift
        # For strict mode - wants resource presence in the required quantity
        # For non-strict mode - minimize deficit (=delta) penalty for the missed resources
        for d in range(self._num_days):
            for s in range(self.num_shifts):
                works = [shifted_resource[n][d][s] for n in range(self.num_resource)]
                # print(works)
                # exit()
                demand = self.required_resources[self.shifts[s]][d]

                if self.strict_mode:
                    sch_model.Add(sum(works) >= demand)
                else:
                    epsilon = sch_model.NewIntVar(0, self.num_resource, f'delta_d{d}s{s}')
                    sch_model.Add(sum(works) >= self.required_resources[self.shifts[s]][d] - epsilon)
                    sch_model.Add(sum(works) <= self.required_resources[self.shifts[s]][d] + epsilon)

                    objective_int_vars.append(epsilon)
                    objective_int_coeffs.append(1)


        # # A resource can at most, work 1 shift per day
        # AD: this is to be covered by shift constraint and dedicated IntVar (0,1)
        for n in range(self.num_resource):
            for d in range(self._num_days):
                sch_model.Add(sum(shifted_resource[n][d][s] for s in range(self.num_shifts)) <= 1)

        # The number of days that an resource rest is not greater that the max allowed
        if self.max_resting > 0:
            working_days = self._num_days - self.max_resting
            for n in range(self.num_resource):
                sch_model.Add(
                    sum(shifted_resource[n][d][s] for d in range(self._num_days) for s in range(self.num_shifts))
                    >= working_days)

        # Create bool matrix of shifts dependencies
        self.non_sequential_shifts_indices = np.zeros(shape=(self.num_shifts, self.num_shifts), dtype='object')
        if self.non_sequential_shifts:
            for dependence in self.non_sequential_shifts:
                i_idx = self.shifts.index(dependence['origin'])
                j_idx = self.shifts.index(dependence['destination'])
                self.non_sequential_shifts_indices[i_idx][j_idx] = 1

        # An resource can not have two consecutive shifts according to shifts dependencies

        for n in range(self.num_resource):
            for d in range(self._num_days - 1):
                for s in range(self.num_shifts):
                    sch_model.Add(
                        sum(shifted_resource[n][d][s] * self.non_sequential_shifts_indices[s][j] +
                            shifted_resource[n][d + 1][j]
                            for j in range(self.num_shifts)) <= 1)

        # resource can't be assigned to banned shifts
        if self.banned_shifts is not None:
            for ban in self.banned_shifts:
                resource_idx = self.resources.index(ban['resource'])
                shift_idx = self.shifts.index(ban['shift'])
                day_idx = int(ban['day'])
                sch_model.Add(shifted_resource[resource_idx][day_idx][shift_idx] == 0)

        # Minimum working hours per resource in the horizon
        # AD: this is replaced by max_shifts_count
        if self.min_working_hours > 0:
            for n in range(self.num_resource):
                sch_model.Add(
                    sum(shifted_resource[n][d][s] * self.shifts_hours[s]
                        for d in range(self._num_days) for s in range(self.num_shifts)) >= self.min_working_hours)

        # max number of shifts, todo: fix it
        if self.max_shifts_count > 0:
            for n in range(self.num_resource):
                works = [shifted_resource[n][d][s] for d in range(self._num_days) for s in range(self.num_shifts)]
                sch_model.Add(
                    sum(works) == self.max_shifts_count)

        # resource shifts preferences

        self.resources_shifts_preferences = np.zeros(shape=(self.num_resource, self.num_shifts), dtype='object')

        if self.resources_preferences:
            for preference in self.resources_preferences:
                resource_idx = self.resources.index(preference['resource'])
                shift_idx = self.shifts.index(preference['shift'])
                self.resources_shifts_preferences[resource_idx][shift_idx] = 1

        # resource relative weight for shift preferences
        self.resources_shifts_weight = np.ones(shape=self.num_resource, dtype='object')
        if self.resources_prioritization:
            for prioritization in self.resources_prioritization:
                resource_idx = self.resources.index(prioritization['resource'])
                self.resources_shifts_weight[resource_idx] = prioritization['weight']

        # Resource shift constraints -- for all employees are same (per day)
        # 1. First we create a new matrix which represents resource working per day

        # daily_resource: 1 if resource is working on day d (sum(shifts) == 1)
        daily_resource = np.empty(shape=(self.num_resource, self._num_days), dtype='object')
        for n in range(self.num_resource):
            for d in range(self._num_days):
                daily_resource[n][d] = sch_model.NewIntVar(0, 1, f'resource_day_n{n}d{d}')

        # 2. Apply constraint on daily work - no more than 1 shift per day
        for n in range(self.num_resource):
            for d in range(self._num_days):
                sch_model.Add(
                    sum(shifted_resource[n][d][s] for s in range(self.num_shifts)) == daily_resource[n][d]
                )

        # 3. Apply sequence constraints of resource daily work
        for ct in self.shift_constraints:
            (hard_min, soft_min, min_cost, soft_max, hard_max, max_cost) = ct
            for n in range(self.num_resource):
                works = [daily_resource[n,d] for d in range(self._num_days)]
                variables, coeffs = add_soft_sequence_constraint(
                    sch_model, works,
                    hard_min, soft_min, min_cost, soft_max, hard_max, max_cost,
                    f'resource_shifts_constraint_n{n}'
                )
                objective_bool_vars.extend(variables)
                objective_bool_coeffs.extend(coeffs)


        # Objective function: Minimize the total number of shifted hours rewarded by resource preferences

        sch_model.Minimize(
            sum(shifted_resource[n][d][s] * (self.shifts_hours[s]
                                             - self.resources_shifts_weight[n] *
                                             self.resources_shifts_preferences[n][s])
                for n in range(self.num_resource)
                for d in range(self._num_days)
                for s in range(self.num_shifts))
            +
            sum(objective_int_vars[i] * objective_int_coeffs[i] for i in range(len(objective_int_vars)))
            +
            sum(objective_bool_vars[i] * objective_bool_coeffs[i] for i in range(len(objective_bool_vars)))
        )

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.max_search_time
        self.solver.num_search_workers = self.num_search_workers

        solution_printer = cp_model.ObjectiveSolutionPrinter()
        self._status = self.solver.Solve(sch_model, solution_printer)

        # Output

        # print('Penalties:')
        # for i, var in enumerate(objective_int_vars):
        #     if self.solver.Value(var) > 0:
        #         print(f'  {var.Name()} violated by {self.solver.Value(var)}, linear penalty={objective_int_coeffs[i]}')

        if self._status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            resource_shifts = []
            resting_resource = []
            shifted_hours = 0
            for n in range(self.num_resource):
                for d in range(self._num_days):
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
