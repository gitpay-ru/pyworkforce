from ortools.sat.python import cp_model

num_employees = 5
num_days = 31
# Creates the model.
model = cp_model.CpModel()

# Creates the variables. employees x days
work = {}
for e in range(num_employees):
    for d in range(num_days):
        work[e, d] = model.NewIntVar(0, 1, f'work_at_day_{d}')

holidays = [model.NewBoolVar(f'holidays_at_day_{d}') for d in range(num_days)]

for e in range(num_employees):
    works = [work[e, d] for d in range(num_days)]
    model.Add(sum(works) == 21)

for d in range(num_days):
    works = [work[e, d] for e in range(num_employees)]
    # model.AddExactlyOne(works + [holidays[d]])
    model.Add(sum(works) > 0).OnlyEnforceIf(holidays[d].Not())
    model.Add(sum(works) == 0).OnlyEnforceIf(holidays[d])

# at least 1 worker should rest
for d in range(num_days):
    not_works = [work[e, d].Not() for e in range(num_employees)]
    model.AddAtLeastOne(not_works)

# no more common 2 holidays in row
num_holidays_in_row = 2
for start in range(len(holidays) - num_holidays_in_row):
    model.AddBoolOr(
        [holidays[i].Not() for i in range(start, start + num_holidays_in_row + 1)])

# at least 4 holidays per month
model.Add(sum(holidays) >= 4)

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)

print(f"Status = {status}")

# Print solution
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    holidays_row = ''
    for d in range(num_days):
        if solver.BooleanValue(holidays[d]):
            holidays_row += u' ■ '
        else:
            holidays_row += u' □ '
    print(f'Holidays: {holidays_row}')

    for e in range(num_employees):
        row = ''
        for d in range(num_days):
            if solver.BooleanValue(work[e, d]):
                row += u' ■ '
            else:
                row += u' □ '
        print(f'Worker {e}: {row}')