from collections import defaultdict


class SolverParams:
    def __init__(self,
                 max_iteration_search_time,
                 solution_limit,
                 num_search_workers,
                 do_logging
                 ):
        self.max_iteration_search_time: float = max_iteration_search_time
        self.solution_limit: int = solution_limit
        self.num_search_workers: int = num_search_workers
        self.do_logging = do_logging


    @staticmethod
    def default():
        return defaultdict(lambda: SolverParams(max_iteration_search_time=300.0,
                                                solution_limit=None,
                                                do_logging=False,
                                                num_search_workers=0))

    @staticmethod
    def from_json(params_dict: any):
        max_iteration_search_time = params_dict.get('max_iteration_search_time', None)
        solution_limit = params_dict.get('solution_limit', None)
        do_logging = params_dict.get('logging', False)
        num_search_workers = params_dict.get('num_search_workers', None)

        return SolverParams(max_iteration_search_time = max_iteration_search_time,
                            solution_limit=solution_limit,
                            do_logging = do_logging,
                            num_search_workers = num_search_workers)