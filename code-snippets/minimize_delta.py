from ortools.sat.python import cp_model

target = 20  # workers

# want to test delta logic
workers_availability = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 100),
    (15, 25)
]
# Creates the model.
model = cp_model.CpModel()

works = {}
delta = {}
delta_sq = {}
delta_positive = {}
# obj_vars = []

for i, (w_min, w_max) in enumerate(workers_availability):
    max_delta = max(target, w_max)

    works[i] = model.NewIntVar(w_min, w_max, f'# of workers (i_{i})')

    delta[i] = model.NewIntVar(0, max_delta, f'delta_i{i}')
    model.Add(works[i] >= target - delta[i])
    model.Add(works[i] <= target + delta[i])

    delta_sq[i] = model.NewIntVar(0, max_delta * max_delta, f'delta_sq_i{i}')
    model.AddMultiplicationEquality(delta_sq[i], [delta[i], delta[i]])

    delta_signed = model.NewIntVar(-max_delta, max_delta, f'delta_signed_i{i}')
    model.Add(delta_signed == works[i] - target)
    delta_positive[i] = model.NewIntVar(0, max_delta, f'delta_positive_i{i}')
    # model.AddAbsEquality(delta_positive[i], delta_signed).OnlyEnforceIf(delta_signed > 0)
    model.AddMaxEquality(delta_positive[i], [delta_signed, 0])

model.Minimize(sum(delta))
# model.Minimize(sum(delta_sq))

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)

print(f"Status = {status}")

# Print solution
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    for i, _ in enumerate(workers_availability):
        print(f'target = {target}, '
              f'works[{i}] = {solver.Value(works[i])}, '
              f'delta = {solver.Value(delta[i])}, sq = {solver.Value(delta_sq[i])}, pos = {solver.Value(delta_positive[i])}')
