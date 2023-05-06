import json
from pyworkforce.rostering.binary_programming import MinHoursRoster
from pyworkforce.utils.solver_params import SolverParams

# scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

with open('scheduling_output_rostering_input_x_4_9_06-00_12-45_15.json', 'r') as f:
    shifts_info = json.load(f)

 # constraint:
#   (hard_min, soft_min, penalty, soft_max, hard_max, penalty)
work_constraints = [
    # work at least 'work_min', but no more than 'work_max',
    # 'work_max' is both lower and upper soft intertval -> deltas are penalized by 1
    (1, 1, 0, 5, 5, 0)
]

rest_constraints = [
    # 1 to 3 non penalized holidays
    (1, 1, 0, 3, 3, 0)
]

num_resources = int(shifts_info["num_resources"])

resources = [f'#{i}' for i in range(num_resources)]
resources_min_w_hours = [176 for _ in range(num_resources)]
resources_max_w_hours = [176 for _ in range(num_resources)]
shifts_hours = [int(i.split('_')[1]) - 1.0 for i in shifts_info["shifts"]]

solver = MinHoursRoster(num_days=shifts_info["num_days"],
                        resources=resources,
                        resources_min_w_hours = resources_min_w_hours,
                        resources_max_w_hours = resources_max_w_hours,
                        shifts=shifts_info["shifts"],
                        shifts_hours=shifts_hours,
                        required_resources=shifts_info["required_resources"],
                        shift_constraints=work_constraints,
                        rest_constraints=rest_constraints,
                        solver_params=SolverParams.default()
                        )


results = solver.solve()
print(results)


# The number of resting resources and shifted resources of all days
# Must match with the list of resources per day
assert results["status"] == 'OPTIMAL'
assert len(results["resting_resource"]) + len(results["resource_shifts"]) == shifts_info["num_resources"]*shifts_info["num_days"]
assert len(results["resource_shifts"]) == results["total_shifts"]
assert results["total_resources"] == shifts_info["num_resources"]
assert results["total_shifts"] == len(results["resource_shifts"])




