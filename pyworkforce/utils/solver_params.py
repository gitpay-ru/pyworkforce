from collections import defaultdict


class SolverParams:
    def __init__(self,
                 max_iteration_search_time,
                 max_iteration_search_time_by_resources,
                 solution_limit,
                 num_search_workers,
                 do_logging
                 ):
        self.max_iteration_search_time: float = max_iteration_search_time
        self.max_iteration_search_time_by_resources: dict = max_iteration_search_time_by_resources
        self.solution_limit: int = solution_limit
        self.num_search_workers: int = num_search_workers
        self.do_logging = do_logging


    @staticmethod
    def default():
        return SolverParams(max_iteration_search_time=600,
                            max_iteration_search_time_by_resources=None,
                            solution_limit=None,
                            do_logging=True,
                            num_search_workers=0)

    @staticmethod
    def from_json(params_dict: any):
        max_iteration_search_time = params_dict.get('max_iteration_search_time', None)
        max_iteration_search_time_by_resources = params_dict.get('max_iteration_search_time_by_resources', None)
        solution_limit = params_dict.get('solution_limit', None)
        do_logging = params_dict.get('logging', False)
        num_search_workers = params_dict.get('num_search_workers', None)

        return SolverParams(max_iteration_search_time = max_iteration_search_time,
                            max_iteration_search_time_by_resources=max_iteration_search_time_by_resources,
                            solution_limit=solution_limit,
                            do_logging = do_logging,
                            num_search_workers = num_search_workers)

    def get_max_iteration_search_time_by_resources(self, num_resources: int):

        DELIMETER = '-'
        DEFAULT = 'default'

        def low_hi(interval_str: str):
            low = int(interval_str.split(DELIMETER)[0])
            hi = int(interval_str.split(DELIMETER)[1])

            return (low, hi)

        if self.max_iteration_search_time_by_resources is not None:
            for k, v in self.max_iteration_search_time_by_resources.items():
                if k == DEFAULT:
                    return v

                (low_incl, hi_incl) = low_hi(k)
                if (low_incl <= num_resources <= hi_incl):
                    return v

        return self.max_iteration_search_time
