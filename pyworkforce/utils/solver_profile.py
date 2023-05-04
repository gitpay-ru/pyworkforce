from pyworkforce.utils.solver_params import SolverParams


class SolverProfile:
    def __init__(self,
                 scheduler_params: SolverParams,
                 roster_params: SolverParams,
                 breaks_params: SolverParams
                 ):
        self._scheduler_params = scheduler_params
        self._roster_params = roster_params
        self._breaks_params = breaks_params

    @property
    def scheduler_params(self):
        return self._scheduler_params

    @property
    def roster_params(self):
        return self._roster_params

    @property
    def breaks_params(self):
        return self._breaks_params

    @staticmethod
    def default():
        return SolverProfile(
            scheduler_params=SolverParams.default(),
            roaster_params=SolverParams.default(),
            breaks_params=SolverParams.default()
        )

    @staticmethod
    def from_json(profile_dict: any):
        scheduler_params = SolverParams.from_json(profile_dict['scheduling'])
        roster_params = SolverParams.from_json(profile_dict['rostering'])
        breaks_params = SolverParams.from_json(profile_dict['breaks'])

        return SolverProfile(
            scheduler_params=scheduler_params,
            roster_params=roster_params,
            breaks_params=breaks_params
        )