from ortools.sat.python import cp_model

num_employees = 3
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
    model.Add(sum(works) == 22)

for d in range(num_days):
    works = [work[e, d] for e in range(num_employees)]
    model.AddBoolXOr(works + [holidays[d]])

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print solution
if status == cp_model.OPTIMAL:

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