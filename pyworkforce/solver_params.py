from collections import defaultdict


class SolverParams:
    def __init__(self,
                 max_iteration_search_time,
                 num_search_workers,
                 do_logging
                 ):
        self.max_iteration_search_time = max_iteration_search_time
        self.num_search_workers = num_search_workers
        self.do_logging = do_logging


    @staticmethod
    def default():
        return defaultdict(lambda: SolverParams(max_iteration_search_time=300.0, do_logging=False, num_search_workers=0))