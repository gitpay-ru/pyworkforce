import json

from pyworkforce.utils.solver_params import SolverParams

profile_str1 = '{' \
               '"logging": true, ' \
               '"max_iteration_search_time_by_resources": {' \
                    '"0-10": 300,' \
                    '"11-50": 600,' \
                    '"51-100": 3600,' \
                    '"default": 10000' \
               '},' \
               '"solution_limit": 20' \
               '}'

profile_str2 = '{' \
               '"logging": true, ' \
               '"max_iteration_search_time": 3600,' \
               '"max_iteration_search_time_by_resources": {' \
                    '"0-10": 300,' \
                    '"11-50": 600' \
               '},' \
               '"solution_limit": 20' \
               '}'

profile_str3 = '{' \
               '"logging": true, ' \
               '"max_iteration_search_time": 3600,' \
               '"solution_limit": 20' \
               '}'


def test_get_max_iteration_search_time_by_resources_1():
    profile_json = json.loads(profile_str1)
    params = SolverParams.from_json(profile_json)

    assert params.get_max_iteration_search_time_by_resources(5) == 300
    assert params.get_max_iteration_search_time_by_resources(50) == 600
    assert params.get_max_iteration_search_time_by_resources(500) == 10000


def test_get_max_iteration_search_time_by_resources_2():
    profile_json = json.loads(profile_str2)
    params = SolverParams.from_json(profile_json)

    assert params.get_max_iteration_search_time_by_resources(5) == 300
    assert params.get_max_iteration_search_time_by_resources(50) == 600
    assert params.get_max_iteration_search_time_by_resources(500) == 3600


def test_get_max_iteration_search_time_by_resources_3():
    profile_json = json.loads(profile_str3)
    params = SolverParams.from_json(profile_json)

    assert params.get_max_iteration_search_time_by_resources(5) == 3600
    assert params.get_max_iteration_search_time_by_resources(50) == 3600
    assert params.get_max_iteration_search_time_by_resources(500) == 3600
