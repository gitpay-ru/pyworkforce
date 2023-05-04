import time

from ortools.sat.python import cp_model


class ObjectiveSolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)

        self.__solution_count = 0
        self.__solution_limit = limit
        self.__start_time = time.time()

    def on_solution_callback(self):
        current_time = time.time()
        print('Solution %i, time = %0.2f s, objective = %i' %
              (self.__solution_count, current_time - self.__start_time, self.ObjectiveValue()))

        self.__solution_count += 1
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count